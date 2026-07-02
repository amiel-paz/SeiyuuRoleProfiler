#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watch_safe_enrichment_llm_tags import source_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export safe-enrichment rows not yet LLM-tagged.")
    parser.add_argument("--input", type=Path, default=Path("data/external/safe_enrichment/character_safe_enrichment.jsonl"))
    parser.add_argument("--tagged", type=Path, default=Path("data/external/safe_enrichment_llm/character_tags.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("data/external/safe_enrichment_llm/errors.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("run/gpu_llm_tagging/pending_safe_enrichment.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("run/gpu_llm_tagging/pending_manifest.json"))
    parser.add_argument("--min-source-chars", type=int, default=80)
    parser.add_argument("--max-source-chars", type=int, default=5000)
    parser.add_argument("--include-errors", action="store_true", help="Retry local parse-error rows on the GPU worker.")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
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


def latest_rows_by_id(rows: list[dict]) -> dict[int, dict]:
    output = {}
    for row in rows:
        character_id = row_id(row)
        if character_id is not None:
            output[character_id] = row
    return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    source_by_id = latest_rows_by_id(read_jsonl(args.input))
    tagged_ids = set(latest_rows_by_id(read_jsonl(args.tagged)))
    error_ids = set(latest_rows_by_id(read_jsonl(args.errors)))
    excluded_ids = set(tagged_ids)
    if not args.include_errors:
        excluded_ids.update(error_ids)

    pending = []
    skipped_no_text = 0
    for character_id, row in source_by_id.items():
        if character_id in excluded_ids:
            continue
        blocks = source_blocks(row, args.max_source_chars)
        total_chars = sum(len(block["text"]) for block in blocks)
        if total_chars < args.min_source_chars:
            skipped_no_text += 1
            continue
        pending.append(row)

    pending.sort(key=lambda row: (-int(row.get("favourites") or 0), str(row.get("name") or "")))
    if args.limit > 0:
        pending = pending[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f"{args.output.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in pending:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.output)

    manifest = {
        "generated_at": utc_now(),
        "source_input": str(args.input),
        "tagged_input": str(args.tagged),
        "errors_input": str(args.errors),
        "output": str(args.output),
        "parameters": {
            "min_source_chars": args.min_source_chars,
            "max_source_chars": args.max_source_chars,
            "include_errors": args.include_errors,
            "limit": args.limit,
        },
        "counts": {
            "safe_enrichment_unique_rows": len(source_by_id),
            "already_tagged": len(tagged_ids),
            "local_errors": len(error_ids),
            "skipped_no_text": skipped_no_text,
            "pending": len(pending),
        },
        "sample_pending": [
            {
                "anilist_character_id": row.get("anilist_character_id"),
                "name": row.get("name"),
                "favourites": row.get("favourites"),
            }
            for row in pending[:20]
        ],
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
