#!/usr/bin/env python3
"""Train from merged HF preference pairs with controllable order and subset-wise replay.

Flow per order:
1) Extract + filter trainable preference rows from HF run artifacts.
2) Build an ordered global list (`subset_desc`, `random`, `reverse`).
3) Split into subset-sized chunks (default uses original subset histogram).
4) For each subset chunk:
   - materialize `artifacts/runs/<run_id>/subsets/subset_XXX/api.jsonl`
   - run `step_subset update-base`
   - optionally run `step_subset eval-ood`
5) Record per-subset latency + throughput metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import validate_config
from scp_stage4.data import write_jsonl
from scp_stage4.schema import validate_artifact_rows


HF_DATASET_REPO = "alwaysgood/scp-stage4-sft-v2-runs"
HF_DATASET_REVISION = "main"
HF_RUNS_ROOT = "artifacts/runs"
_DEFAULT_PRUNE_ORDERS = {"random", "reverse"}
_DEFAULT_BASE_UPDATE_PER_DEVICE_BATCH_SIZE = 4


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _list_run_ids(*, repo_id: str, revision: str, runs_root: str) -> list[str]:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_runs_root = urllib.parse.quote(runs_root, safe="/")
    url = (
        f"https://huggingface.co/api/datasets/{quoted_repo}/tree/"
        f"{quoted_revision}/{quoted_runs_root}?recursive=false&expand=false&limit=1000"
    )
    rows = _fetch_json(url)
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected HF tree payload for {url}: expected list")
    run_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("type") != "directory":
            continue
        path = str(row.get("path", "")).strip()
        if not path:
            continue
        run_ids.append(Path(path).name)
    return sorted(set(run_ids))


def _download_preference_rows(
    *,
    repo_id: str,
    revision: str,
    run_id: str,
) -> list[dict[str, Any]]:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_run_id = urllib.parse.quote(run_id, safe="")
    file_path = f"{HF_RUNS_ROOT}/{quoted_run_id}/preference_pairs.jsonl"
    url = (
        f"https://huggingface.co/datasets/{quoted_repo}/resolve/"
        f"{quoted_revision}/{file_path}?download=true"
    )
    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise

    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, Mapping):
            raise RuntimeError(f"{run_id} preference row {line_no} is not an object")
        row = dict(parsed)
        row["_source_run_id"] = run_id
        rows.append(row)
    return rows


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _get_by_dotpath(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _override_key(raw: str) -> str:
    text = str(raw).strip()
    if "=" not in text:
        return text
    return text.split("=", 1)[0].strip()


def _has_override(overrides: Iterable[str], key: str) -> bool:
    target = str(key).strip()
    if not target:
        return False
    return any(_override_key(raw) == target for raw in overrides)


def _extract_train_examples(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        status = str(row.get("status", "")).strip()
        gold = row.get("gold")
        source = str(row.get("source", "")).strip()
        student = str(row.get("student", "")).strip()
        teacher_label = str(row.get("teacher_label", "")).strip()
        dataset = str(row.get("dataset", "")).strip()
        if status != "ok":
            continue
        if not isinstance(gold, str) or not gold.strip():
            continue
        if not source or not student or not teacher_label or not dataset:
            continue

        source_run = str(row.get("_source_run_id", "")).strip()
        subset_idx = _as_int(row.get("subset_idx", 0), 0)
        orig_id = str(row.get("id", "")).strip()
        row_id = str(row.get("row_id", "")).strip() or orig_id
        request_id = str(row.get("request_id", "")).strip()
        example_uid = request_id or f"{source_run}/pref/{idx:08d}"

        examples.append(
            {
                "example_uid": example_uid,
                "orig_id": orig_id or example_uid,
                "row_id": row_id or example_uid,
                "request_id": request_id,
                "source": source,
                "target": gold,
                "student": student,
                "teacher_label": teacher_label,
                "metadata": _as_dict(row.get("metadata")),
                "dataset": dataset,
                "provider": str(row.get("provider", "")).strip(),
                "model": str(row.get("model", "")).strip(),
                "prompt_version": row.get("prompt_version"),
                "prompt_hash": row.get("prompt_hash"),
                "usage": row.get("usage"),
                "cost": row.get("cost"),
                "latency_ms": row.get("latency_ms"),
                "attempt": row.get("attempt"),
                "error": row.get("error"),
                "config_hash": row.get("config_hash"),
                "split_name": row.get("split_name"),
                "thinking_text": row.get("thinking_text"),
                "source_run_id": source_run,
                "source_subset_idx": subset_idx,
                "source_order_idx": idx,
            }
        )
    return examples


def _order_examples(
    examples: list[dict[str, Any]],
    *,
    order_name: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    if order_name == "reverse":
        return list(reversed(examples))
    if order_name == "subset_desc":
        indexed = list(enumerate(examples))
        indexed.sort(
            key=lambda item: (
                -_as_int(item[1].get("source_subset_idx", 0), 0),
                item[0],
            )
        )
        return [row for _, row in indexed]
    if order_name == "random":
        shuffled = list(examples)
        rng = random.Random(int(random_seed))
        rng.shuffle(shuffled)
        return shuffled
    raise RuntimeError(f"unsupported order: {order_name}")


def _reference_subset_plan(examples: list[Mapping[str, Any]]) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    for row in examples:
        idx = _as_int(row.get("source_subset_idx", 0), 0)
        counts[idx] = counts.get(idx, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0], reverse=True)


def _split_by_plan(
    ordered_examples: list[dict[str, Any]],
    plan: list[tuple[int, int]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    total = sum(count for _, count in plan)
    if total != len(ordered_examples):
        raise RuntimeError(
            f"subset plan count mismatch: plan_total={total} ordered={len(ordered_examples)}"
        )
    out: list[tuple[int, list[dict[str, Any]]]] = []
    cursor = 0
    for subset_idx, count in plan:
        nxt = cursor + int(count)
        out.append((int(subset_idx), ordered_examples[cursor:nxt]))
        cursor = nxt
    return out


def _resolve_primary_provider_model(cfg: Mapping[str, Any]) -> tuple[str, str]:
    primary = _as_dict(_as_dict(cfg.get("external_api")).get("primary"))
    provider = str(primary.get("provider", "unknown_provider")).strip() or "unknown_provider"
    model = str(primary.get("model", "unknown_model")).strip() or "unknown_model"
    return provider, model


def _checkpoint_keep_last_n(cfg: Mapping[str, Any]) -> int:
    raw = _get_by_dotpath(cfg, "training.checkpoint.keep_last_n", 1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 1
    return max(1, int(raw))


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _prune_old_subset_train_finals(
    *,
    run_root: Path,
    processed_subset_indices: list[int],
    keep_last_n: int,
) -> dict[str, Any]:
    keep_n = max(1, int(keep_last_n))
    preserve = set(processed_subset_indices[-keep_n:])
    deleted_subset_indices: list[int] = []
    deleted_count = 0
    freed_bytes = 0

    for subset_idx in processed_subset_indices[:-keep_n]:
        subset_dir = run_root / "subsets" / f"subset_{subset_idx:03d}"
        train_final_dir = subset_dir / "train_final"
        if not train_final_dir.exists():
            continue
        freed_bytes += _path_size_bytes(train_final_dir)
        shutil.rmtree(train_final_dir, ignore_errors=True)
        deleted_count += 1
        deleted_subset_indices.append(int(subset_idx))

    return {
        "keep_last_n": keep_n,
        "preserved_subset_indices": sorted(int(x) for x in preserve),
        "deleted_subset_indices": deleted_subset_indices,
        "deleted_count": int(deleted_count),
        "freed_bytes": int(freed_bytes),
    }


def _build_api_rows(
    *,
    subset_examples: list[dict[str, Any]],
    run_id: str,
    subset_idx: int,
    default_provider: str,
    default_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_idx, ex in enumerate(subset_examples):
        uid = str(ex.get("example_uid", "")).strip() or f"row_{local_idx:08d}"
        request_id = f"{run_id}/subsets/subset_{subset_idx:03d}/{local_idx:06d}/api"
        provider = str(ex.get("provider", "")).strip() or default_provider
        model = str(ex.get("model", "")).strip() or default_model
        rows.append(
            {
                "id": uid,
                "row_id": uid,
                "dataset": str(ex["dataset"]),
                "source": str(ex["source"]),
                "metadata": _as_dict(ex.get("metadata")),
                "request_id": request_id,
                "run_id": run_id,
                "subset_idx": int(subset_idx),
                "provider": provider,
                "model": model,
                "status": "ok",
                "teacher_label": str(ex["teacher_label"]),
                "student": str(ex["student"]),
                "gold": str(ex["target"]),
                "reason": None,
                "prompt_version": ex.get("prompt_version"),
                "prompt_hash": ex.get("prompt_hash"),
                "usage": ex.get("usage"),
                "cost": ex.get("cost"),
                "latency_ms": ex.get("latency_ms"),
                "attempt": ex.get("attempt"),
                "error": ex.get("error"),
                "config_hash": ex.get("config_hash"),
                "split_name": ex.get("split_name"),
                "thinking_text": ex.get("thinking_text"),
            }
        )
    return validate_artifact_rows(rows, "api")


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_step_subset_phase(
    *,
    command: str,
    config_path: str,
    run_id: str,
    subset_idx: int,
    overrides: list[str],
    require_checkpoint: bool,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scp_stage4.pipeline.step_subset",
        command,
        "--config",
        config_path,
        "--run-id",
        run_id,
        "--subset-idx",
        str(subset_idx),
    ]
    effective_overrides = list(overrides)
    if require_checkpoint:
        effective_overrides.append("training.base_update.requires_base_checkpoint=true")
    cmd.extend(effective_overrides)
    _run(cmd, cwd=Path.cwd())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=HF_DATASET_REPO)
    parser.add_argument("--revision", default=HF_DATASET_REVISION)
    parser.add_argument("--run-ids", nargs="*", default=None)
    parser.add_argument("--config", default="configs/scp_stage4_real.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--run-id-prefix", default="pref_order")
    parser.add_argument("--orders", nargs="+", default=["subset_desc", "random"])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--clean-out-dir", action="store_true")
    parser.add_argument("--clean-run-root", action="store_true")
    parser.add_argument("--eval-after-each-subset", action="store_true", default=True)
    parser.add_argument("--no-eval-after-each-subset", action="store_false", dest="eval_after_each_subset")
    parser.add_argument(
        "--eval-force-vllm-subprocess",
        action="store_true",
        default=True,
        help=(
            "When running eval-ood, force inference.runtime.vllm_inprocess.enabled=false "
            "so vLLM exits before COMET/xCOMET scoring."
        ),
    )
    parser.add_argument(
        "--no-eval-force-vllm-subprocess",
        action="store_false",
        dest="eval_force_vllm_subprocess",
    )
    parser.add_argument(
        "--prune-train-final-orders",
        nargs="*",
        default=sorted(_DEFAULT_PRUNE_ORDERS),
        help="Delete older subset train_final dirs only for these orders.",
    )
    args, overrides = parser.parse_known_args(argv)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("artifacts") / "experiments" / f"{args.run_id_prefix}_{_utc_now_stamp()}"
    )
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_overrides = list(overrides)
    if not _has_override(
        effective_overrides,
        "training.base_update.batching.per_device_train_batch_size",
    ):
        effective_overrides.append(
            "training.base_update.batching.per_device_train_batch_size="
            f"{_DEFAULT_BASE_UPDATE_PER_DEVICE_BATCH_SIZE}"
        )

    cfg = compose_config(args.config, overrides=effective_overrides)
    validate_config(cfg)

    run_ids = list(args.run_ids) if args.run_ids else _list_run_ids(
        repo_id=args.repo_id, revision=args.revision, runs_root=HF_RUNS_ROOT
    )
    if not run_ids:
        raise SystemExit("no run ids found")

    all_preference_rows: list[dict[str, Any]] = []
    per_run_counts: dict[str, dict[str, int]] = {}
    for run_id in run_ids:
        rows = _download_preference_rows(repo_id=args.repo_id, revision=args.revision, run_id=run_id)
        ok_rows = sum(1 for row in rows if str(row.get("status", "")).strip() == "ok")
        per_run_counts[run_id] = {"preference_rows": len(rows), "ok_rows": ok_rows}
        all_preference_rows.extend(rows)
        print(
            f"[extract] run_id={run_id} preference_rows={len(rows)} ok_rows={ok_rows}",
            file=sys.stderr,
            flush=True,
        )

    merged_pref_path = out_dir / "preference_pairs.merged.jsonl"
    write_jsonl(merged_pref_path, all_preference_rows, ensure_ascii=False)

    examples = _extract_train_examples(all_preference_rows)
    if args.max_examples is not None:
        examples = examples[: max(0, int(args.max_examples))]
    if not examples:
        raise SystemExit("no trainable examples (status=ok with required fields)")

    write_jsonl(out_dir / "train_examples.base.jsonl", examples, ensure_ascii=False)

    reference_plan = _reference_subset_plan(examples)
    if not reference_plan:
        raise SystemExit("empty subset plan")

    default_provider, default_model = _resolve_primary_provider_model(cfg)

    summary: dict[str, Any] = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "config": args.config,
        "config_overrides": effective_overrides,
        "orders": args.orders,
        "random_seed": int(args.random_seed),
        "run_ids": run_ids,
        "per_run_counts": per_run_counts,
        "total_preference_rows": len(all_preference_rows),
        "total_train_examples": len(examples),
        "prepare_only": bool(args.prepare_only),
        "eval_after_each_subset": bool(args.eval_after_each_subset),
        "eval_force_vllm_subprocess": bool(args.eval_force_vllm_subprocess),
        "prune_train_final_orders": list(args.prune_train_final_orders),
        "reference_subset_plan": [
            {"subset_idx": int(subset_idx), "count": int(count)}
            for subset_idx, count in reference_plan
        ],
        "outputs": {},
    }

    for order_name in args.orders:
        ordered = _order_examples(examples, order_name=order_name, random_seed=int(args.random_seed))
        chunked = _split_by_plan(ordered, reference_plan)

        order_dir = out_dir / order_name
        order_dir.mkdir(parents=True, exist_ok=True)
        ordered_examples_path = order_dir / f"train_examples.{order_name}.jsonl"
        write_jsonl(ordered_examples_path, ordered, ensure_ascii=False)

        run_id = f"{args.run_id_prefix}_{order_name}_{_utc_now_stamp()}"
        run_root = Path("artifacts") / "runs" / run_id
        if run_root.exists():
            if args.clean_run_root:
                shutil.rmtree(run_root)
            else:
                raise RuntimeError(
                    f"run root already exists: {run_root}. "
                    "Use --clean-run-root or a different --run-id-prefix."
                )

        subset_summaries: list[dict[str, Any]] = []
        processed_subset_indices: list[int] = []
        local_keep_last_n = _checkpoint_keep_last_n(cfg)
        for pos, (subset_idx, subset_examples) in enumerate(chunked):
            subset_root = run_root / "subsets" / f"subset_{subset_idx:03d}"
            subset_root.mkdir(parents=True, exist_ok=True)
            api_path = subset_root / "api.jsonl"
            api_rows = _build_api_rows(
                subset_examples=subset_examples,
                run_id=run_id,
                subset_idx=subset_idx,
                default_provider=default_provider,
                default_model=default_model,
            )
            write_jsonl(api_path, api_rows, ensure_ascii=False)

            train_seconds = 0.0
            train_rows = len(api_rows)
            checkpoint_path = None
            eval_seconds = 0.0
            eval_rows = 0
            eval_summary_path = None

            if not args.prepare_only:
                train_t0 = time.perf_counter()
                _run_step_subset_phase(
                    command="update-base",
                    config_path=args.config,
                    run_id=run_id,
                    subset_idx=subset_idx,
                    overrides=effective_overrides,
                    require_checkpoint=(pos > 0),
                )
                train_seconds = time.perf_counter() - train_t0

                checkpoint_state_path = subset_root / "train_final" / "checkpoint_state.json"
                if checkpoint_state_path.exists():
                    checkpoint_state = _read_json(checkpoint_state_path)
                    checkpoint_path = checkpoint_state.get("checkpoint_path")

                if args.eval_after_each_subset:
                    eval_overrides = list(effective_overrides)
                    if args.eval_force_vllm_subprocess:
                        eval_overrides.append("inference.runtime.vllm_inprocess.enabled=false")
                    eval_t0 = time.perf_counter()
                    _run_step_subset_phase(
                        command="eval-ood",
                        config_path=args.config,
                        run_id=run_id,
                        subset_idx=subset_idx,
                        overrides=eval_overrides,
                        require_checkpoint=False,
                    )
                    eval_seconds = time.perf_counter() - eval_t0

                    summary_path = (
                        run_root
                        / "eval"
                        / "ood_test"
                        / f"subset_{subset_idx:03d}.summary.json"
                    )
                    if summary_path.exists():
                        ood_summary = _read_json(summary_path)
                        eval_rows = _as_int(ood_summary.get("rows", 0), 0)
                        eval_summary_path = str(summary_path)

                processed_subset_indices.append(int(subset_idx))
                prune_stats = None
                if order_name in set(args.prune_train_final_orders):
                    prune_stats = _prune_old_subset_train_finals(
                        run_root=run_root,
                        processed_subset_indices=processed_subset_indices,
                        keep_last_n=local_keep_last_n,
                    )
            else:
                prune_stats = None

            subset_summary = {
                "subset_idx": int(subset_idx),
                "rows": int(train_rows),
                "api_path": str(api_path),
                "checkpoint_path": checkpoint_path,
                "train_seconds": train_seconds,
                "train_rows_per_sec": (float(train_rows) / train_seconds) if train_seconds > 0 else None,
                "eval_seconds": eval_seconds,
                "eval_rows": int(eval_rows),
                "eval_rows_per_sec": (float(eval_rows) / eval_seconds) if eval_seconds > 0 else None,
                "eval_summary_path": eval_summary_path,
                "local_prune": prune_stats,
            }
            subset_summaries.append(subset_summary)

        total_train_rows = sum(item["rows"] for item in subset_summaries)
        total_train_seconds = sum(float(item["train_seconds"]) for item in subset_summaries)
        total_eval_rows = sum(int(item["eval_rows"]) for item in subset_summaries)
        total_eval_seconds = sum(float(item["eval_seconds"]) for item in subset_summaries)

        summary["outputs"][order_name] = {
            "run_id": run_id,
            "run_root": str(run_root),
            "ordered_examples_path": str(ordered_examples_path),
            "executed": not bool(args.prepare_only),
            "subset_count": len(subset_summaries),
            "subsets": subset_summaries,
            "throughput": {
                "train_total_rows": total_train_rows,
                "train_total_seconds": total_train_seconds,
                "train_rows_per_sec": (float(total_train_rows) / total_train_seconds)
                if total_train_seconds > 0
                else None,
                "eval_total_rows": total_eval_rows,
                "eval_total_seconds": total_eval_seconds,
                "eval_rows_per_sec": (float(total_eval_rows) / total_eval_seconds)
                if total_eval_seconds > 0
                else None,
            },
        }

    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({"status": "ok", "summary_path": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
