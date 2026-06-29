#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cache_llm_character_tags import canonical_tag
from fit_topics import import_nltk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LLM description tags with canonical non-role VNDB traits.")
    parser.add_argument("--llm-tags-input", type=Path, required=True)
    parser.add_argument("--vndb-tags-input", type=Path, default=Path("data/external/vndb/vndb_character_traits.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-vndb-groups", nargs="*", default=None)
    parser.add_argument("--exclude-vndb-groups", nargs="*", default=["Role"])
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalized_group_set(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {value.strip().lower() for value in values}


def empty_tags() -> dict[str, list[dict]]:
    return {"role": [], "personality": [], "traits": []}


def add_tag(tags: dict[str, list[dict]], category: str, tag: dict) -> None:
    value = tag["tag"]
    for existing in tags[category]:
        if existing["tag"] == value:
            existing_sources = set(existing.get("sources", []))
            existing_sources.update(tag.get("sources", []))
            existing["sources"] = sorted(existing_sources)
            if tag.get("vndb_groups"):
                existing["vndb_groups"] = sorted(set(existing.get("vndb_groups", []) + tag["vndb_groups"]))
            if tag.get("evidence"):
                existing.setdefault("evidence", [])
                existing["evidence"].extend(
                    evidence for evidence in tag["evidence"] if evidence not in existing["evidence"]
                )
            return
    tags[category].append(tag)


def llm_tag_payload(tag: dict) -> dict:
    return {
        "tag": tag["tag"],
        "confidence": tag.get("confidence", ""),
        "sources": ["llm"],
        "evidence": [tag["evidence"]] if tag.get("evidence") else [],
    }


def vndb_category(group: str) -> str:
    return "personality" if group == "Personality" else "traits"


def merge_character(
    llm_character: dict,
    vndb_character: dict | None,
    nltk: Any,
    include_groups: set[str] | None,
    exclude_groups: set[str],
) -> dict:
    tags = empty_tags()
    for category in ("personality", "traits"):
        for tag in llm_character.get("llm_tags", {}).get(category, []):
            add_tag(tags, category, llm_tag_payload(tag))

    rejected_vndb = []
    accepted_vndb = []
    if vndb_character:
        match = vndb_character.get("vndb_best_match") or {}
        for trait in match.get("traits") or []:
            group = str(trait.get("group") or "")
            group_key = group.lower()
            if group_key in exclude_groups:
                rejected_vndb.append({"name": trait.get("name"), "group": group, "reason": "excluded_group"})
                continue
            if include_groups is not None and group_key not in include_groups:
                rejected_vndb.append({"name": trait.get("name"), "group": group, "reason": "not_included_group"})
                continue
            tag_value = canonical_tag(str(trait.get("name") or ""), nltk)
            if not tag_value:
                rejected_vndb.append({"name": trait.get("name"), "group": group, "reason": "non_canonical_phrase"})
                continue
            payload = {
                "tag": tag_value,
                "confidence": "vndb",
                "sources": ["vndb"],
                "evidence": [],
                "vndb_groups": [group],
                "vndb_char_count": trait.get("char_count", 0),
            }
            category = vndb_category(group)
            add_tag(tags, category, payload)
            accepted_vndb.append({"tag": tag_value, "group": group})

    return {
        **llm_character,
        "llm_tags": tags,
        "merged_descriptor_sources": {
            "accepted_vndb": accepted_vndb,
            "rejected_vndb": rejected_vndb,
            "vndb_character_id": (vndb_character.get("vndb_best_match") or {}).get("vndb_character_id")
            if vndb_character
            else None,
        },
    }


def main() -> None:
    args = parse_args()
    nltk = import_nltk()
    llm_payload = read_json(args.llm_tags_input)
    vndb_payload = read_json(args.vndb_tags_input)
    vndb_by_id = {int(row["anilist_character_id"]): row for row in vndb_payload.get("characters", [])}
    include_groups = normalized_group_set(args.include_vndb_groups)
    exclude_groups = normalized_group_set(args.exclude_vndb_groups) or set()

    characters = [
        merge_character(
            character,
            vndb_by_id.get(int(character["anilist_character_id"])),
            nltk,
            include_groups,
            exclude_groups,
        )
        for character in llm_payload.get("characters", [])
    ]

    source_counts = Counter()
    vndb_group_counts = Counter()
    descriptor_counts = {"personality": Counter(), "traits": Counter()}
    for character in characters:
        for category in ("personality", "traits"):
            for tag in character.get("llm_tags", {}).get(category, []):
                descriptor_counts[category][tag["tag"]] += 1
                for source in tag.get("sources", []):
                    source_counts[source] += 1
                for group in tag.get("vndb_groups", []):
                    vndb_group_counts[group] += 1

    payload = {
        "generated_at": utc_now(),
        "source": "merge_llm_vndb_character_tags.py",
        "parameters": {
            "llm_tags_input": str(args.llm_tags_input),
            "vndb_tags_input": str(args.vndb_tags_input),
            "include_vndb_groups": args.include_vndb_groups,
            "exclude_vndb_groups": args.exclude_vndb_groups,
            "vndb_rule": "keep VNDB traits only when group is not excluded and trait name canonicalizes to nouns/adjectives",
        },
        "counts": {
            "characters": len(characters),
            "characters_with_vndb_match": sum(
                1 for character in characters if character["merged_descriptor_sources"]["vndb_character_id"]
            ),
            "source_tag_counts": dict(source_counts),
            "vndb_group_counts": dict(vndb_group_counts),
            "personality_descriptors": len(descriptor_counts["personality"]),
            "trait_descriptors": len(descriptor_counts["traits"]),
        },
        "descriptor_counts": {
            category: descriptor_counts[category].most_common() for category in ("personality", "traits")
        },
        "characters": characters,
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
