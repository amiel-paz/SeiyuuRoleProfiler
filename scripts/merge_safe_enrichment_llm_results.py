#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GPU safe-enrichment LLM results into local JSONL caches.")
    parser.add_argument("--local-tags", type=Path, default=Path("data/external/safe_enrichment_llm/character_tags.jsonl"))
    parser.add_argument("--local-errors", type=Path, default=Path("data/external/safe_enrichment_llm/errors.jsonl"))
    parser.add_argument("--local-raw-dir", type=Path, default=Path("data/external/safe_enrichment_llm/raw"))
    parser.add_argument("--remote-tags", type=Path, required=True)
    parser.add_argument("--remote-errors", type=Path, default=None)
    parser.add_argument("--remote-raw-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("run/gpu_llm_tagging/merge_report.json"))
    parser.add_argument("--replace-errors-with-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def row_id(row: dict) -> int | None:
    value = row.get("anilist_character_id") or row.get("character_id") or row.get("id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def latest_by_id(rows: list[dict]) -> dict[int, dict]:
    output = {}
    for row in rows:
        character_id = row_id(row)
        if character_id is not None:
            output[character_id] = row
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def copy_raw_cache(remote_raw_dir: Path | None, local_raw_dir: Path, character_ids: set[int], dry_run: bool) -> int:
    if remote_raw_dir is None or not remote_raw_dir.exists():
        return 0
    copied = 0
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    for character_id in sorted(character_ids):
        source = remote_raw_dir / f"{character_id}.json"
        target = local_raw_dir / f"{character_id}.json"
        if not source.exists() or target.exists():
            continue
        copied += 1
        if not dry_run:
            shutil.copy2(source, target)
    return copied


def main() -> None:
    args = parse_args()
    local_tag_rows = read_jsonl(args.local_tags)
    local_error_rows = read_jsonl(args.local_errors)
    remote_tag_rows = read_jsonl(args.remote_tags)
    remote_error_rows = read_jsonl(args.remote_errors)

    local_tags = latest_by_id(local_tag_rows)
    local_errors = latest_by_id(local_error_rows)
    remote_tags = latest_by_id(remote_tag_rows)
    remote_errors = latest_by_id(remote_error_rows)

    new_success_ids = set(remote_tags) - set(local_tags)
    if not args.replace_errors_with_success:
        new_success_ids -= set(local_errors)
    new_error_ids = set(remote_errors) - set(local_errors) - set(local_tags)

    merged_tags = latest_by_id(local_tag_rows)
    for character_id in sorted(new_success_ids):
        merged_tags[character_id] = remote_tags[character_id]

    merged_errors = latest_by_id(local_error_rows)
    if args.replace_errors_with_success:
        for character_id in new_success_ids:
            merged_errors.pop(character_id, None)
    for character_id in sorted(new_error_ids):
        merged_errors[character_id] = remote_errors[character_id]

    raw_copied = copy_raw_cache(args.remote_raw_dir, args.local_raw_dir, new_success_ids, args.dry_run)

    report = {
        "generated_at": utc_now(),
        "dry_run": args.dry_run,
        "inputs": {
            "local_tags": str(args.local_tags),
            "local_errors": str(args.local_errors),
            "remote_tags": str(args.remote_tags),
            "remote_errors": str(args.remote_errors) if args.remote_errors else "",
            "remote_raw_dir": str(args.remote_raw_dir) if args.remote_raw_dir else "",
        },
        "counts": {
            "local_success_before": len(local_tags),
            "local_errors_before": len(local_errors),
            "remote_success": len(remote_tags),
            "remote_errors": len(remote_errors),
            "new_successes": len(new_success_ids),
            "new_errors": len(new_error_ids),
            "local_success_after": len(merged_tags),
            "local_errors_after": len(merged_errors),
            "raw_cache_files_copied": raw_copied,
        },
        "new_success_ids_sample": sorted(new_success_ids)[:50],
        "new_error_ids_sample": sorted(new_error_ids)[:50],
    }

    if not args.dry_run:
        write_jsonl(args.local_tags, [merged_tags[key] for key in sorted(merged_tags)])
        write_jsonl(args.local_errors, [merged_errors[key] for key in sorted(merged_errors)])
    write_json(args.report, report)
    print(json.dumps(report["counts"], indent=2))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
