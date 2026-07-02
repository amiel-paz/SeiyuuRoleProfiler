#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATEGORIES = ("role", "personality", "traits")
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
GENERIC_TAGS = {
    "ability",
    "behavior",
    "character",
    "condition",
    "interest",
    "narrative",
    "personality",
    "relationship",
    "role",
    "school",
    "skill",
    "story",
    "trait",
    "traits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggressively normalize and dedupe LLM character tags.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "run/gpu_llm_tagging/a100_batch_transformers_prod_20260701/"
            "batch_transformers_prod_complete/character_tags.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "run/gpu_llm_tagging/a100_batch_transformers_prod_20260701/"
            "batch_transformers_prod_complete/character_tags_deduped_aggressive.jsonl"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "run/gpu_llm_tagging/a100_batch_transformers_prod_20260701/"
            "batch_transformers_prod_complete/dedupe_aggressive_summary.json"
        ),
    )
    parser.add_argument("--keep-generic", action="store_true")
    parser.add_argument("--keep-non-english", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_tag(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("_", " ").replace("/", " ")
    value = value.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"[`'\"“”‘’]+", "", value)
    value = re.sub(r"[^0-9A-Za-z -]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def is_englishish(value: str) -> bool:
    return bool(value) and all(ord(char) < 128 for char in value)


def singularize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def token_key(value: str) -> tuple[str, ...]:
    return tuple(singularize_token(token) for token in value.split() if token)


def better_item(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_tag = left["tag"]
    right_tag = right["tag"]
    left_score = (CONFIDENCE_RANK.get(left.get("confidence"), 0), len(token_key(left_tag)), len(left_tag))
    right_score = (CONFIDENCE_RANK.get(right.get("confidence"), 0), len(token_key(right_tag)), len(right_tag))
    return left if left_score >= right_score else right


def item_for_tag(raw: dict[str, Any], normalized_tag: str) -> dict[str, Any]:
    item = dict(raw)
    original_tag = str(raw.get("tag") or "")
    item["tag"] = normalized_tag
    if original_tag != normalized_tag:
        item["original_tag"] = original_tag
    item.setdefault("aliases", [])
    if original_tag and original_tag != normalized_tag:
        item["aliases"].append(original_tag)
    item["merged_count"] = 1
    return item


def merge_items(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    winner = better_item(target, source)
    loser = source if winner is target else target
    aliases = list(dict.fromkeys((winner.get("aliases") or []) + [loser.get("tag", "")] + (loser.get("aliases") or [])))
    aliases = [alias for alias in aliases if alias and alias != winner.get("tag")]
    winner["aliases"] = aliases
    winner["merged_count"] = int(target.get("merged_count", 1)) + int(source.get("merged_count", 1))
    return winner


def dedupe_category(items: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    normalized_items: list[dict[str, Any]] = []
    for raw in items:
        tag = normalize_tag(str(raw.get("tag") or ""))
        if not tag:
            stats["dropped_empty_after_normalization"] += 1
            continue
        if not args.keep_non_english and not is_englishish(str(raw.get("tag") or "")):
            stats["dropped_non_english"] += 1
            continue
        if not args.keep_generic and tag in GENERIC_TAGS:
            stats["dropped_generic"] += 1
            continue
        normalized_items.append(item_for_tag(raw, tag))

    by_exact: dict[str, dict[str, Any]] = {}
    for item in normalized_items:
        tag = item["tag"]
        if tag in by_exact:
            by_exact[tag] = merge_items(by_exact[tag], item)
            stats["merged_exact"] += 1
        else:
            by_exact[tag] = item

    by_tokens: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in by_exact.values():
        key = token_key(item["tag"])
        if key in by_tokens:
            by_tokens[key] = merge_items(by_tokens[key], item)
            stats["merged_token_equivalent"] += 1
        else:
            by_tokens[key] = item

    candidates = sorted(
        by_tokens.values(),
        key=lambda item: (
            CONFIDENCE_RANK.get(item.get("confidence"), 0),
            len(token_key(item["tag"])),
            len(item["tag"]),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for item in candidates:
        item_tokens = set(token_key(item["tag"]))
        if not item_tokens:
            continue
        merged = False
        for index, existing in enumerate(kept):
            existing_tokens = set(token_key(existing["tag"]))
            if not existing_tokens:
                continue
            if item_tokens <= existing_tokens or existing_tokens <= item_tokens:
                kept[index] = merge_items(existing, item)
                stats["merged_subset_superset"] += 1
                merged = True
                break
        if not merged:
            kept.append(item)

    kept.sort(key=lambda item: (-CONFIDENCE_RANK.get(item.get("confidence"), 0), item["tag"]))
    return kept, stats


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    output_rows = []
    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "input": str(args.input),
        "output": str(args.output),
        "parameters": {
            "keep_generic": args.keep_generic,
            "keep_non_english": args.keep_non_english,
            "generic_tag_count": len(GENERIC_TAGS),
        },
        "rows": len(rows),
        "categories": {},
        "examples": defaultdict(list),
    }
    total_before = Counter()
    total_after = Counter()
    total_stats = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_name(f"{args.output.name}.tmp").open("w", encoding="utf-8") as handle:
        for row in rows:
            row = dict(row)
            tags = row.get("tags") or {}
            cleaned_tags = {}
            for category in CATEGORIES:
                before_items = tags.get(category) or []
                total_before[category] += len(before_items)
                cleaned, stats = dedupe_category(before_items, args)
                total_after[category] += len(cleaned)
                total_stats.update({f"{category}.{key}": value for key, value in stats.items()})
                cleaned_tags[category] = cleaned
                if len(summary["examples"][category]) < 12 and len(cleaned) != len(before_items):
                    summary["examples"][category].append(
                        {
                            "name": row.get("name"),
                            "before": [item.get("tag") for item in before_items],
                            "after": [item.get("tag") for item in cleaned],
                        }
                    )
            row["tags"] = cleaned_tags
            row["tag_cleaning"] = {
                "dedupe": "aggressive_lexical_v1",
                "generated_at": summary["generated_at"],
            }
            output_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output.with_name(f"{args.output.name}.tmp").replace(args.output)

    for category in CATEGORIES:
        summary["categories"][category] = {
            "before": total_before[category],
            "after": total_after[category],
            "removed": total_before[category] - total_after[category],
            "removed_fraction": round(
                (total_before[category] - total_after[category]) / max(total_before[category], 1),
                4,
            ),
        }
    summary["total"] = {
        "before": sum(total_before.values()),
        "after": sum(total_after.values()),
        "removed": sum(total_before.values()) - sum(total_after.values()),
        "removed_fraction": round(
            (sum(total_before.values()) - sum(total_after.values())) / max(sum(total_before.values()), 1),
            4,
        ),
    }
    summary["operations"] = dict(total_stats)
    summary["empty_rows_after"] = sum(
        1 for row in output_rows if not any(row["tags"].get(category) for category in CATEGORIES)
    )
    summary["examples"] = dict(summary["examples"])
    write_json(args.summary, summary)
    print(json.dumps(summary["total"], indent=2))
    print(json.dumps(summary["categories"], indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
