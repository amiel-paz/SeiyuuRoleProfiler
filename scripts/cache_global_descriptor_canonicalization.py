#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from seiyuu_character_semantic_clusters import (
    canonicalize_descriptors,
    descriptor_tokens,
)
from seiyuu_local_nmf_lane_svd import (
    descriptor_rows_from_character,
    load_or_create_embeddings,
    load_safe_llm_personality,
    read_json,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache a global descriptor canonicalization map over the full character descriptor universe."
    )
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"),
    )
    parser.add_argument(
        "--safe-llm-tags",
        nargs="*",
        type=Path,
        default=[
            Path("data/external/safe_enrichment_llm/character_tags.jsonl"),
            Path("run/gpu_llm_tagging/returned_latest/character_tags.jsonl"),
        ],
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/global_descriptor_canonicalization"),
    )
    parser.add_argument("--similarity-threshold", type=float, default=1.01)
    parser.add_argument("--contained-distance-threshold", type=float, default=0.16)
    parser.add_argument("--max-examples-per-descriptor", type=int, default=5)
    return parser.parse_args()


def character_name(source: dict) -> str:
    name = source.get("name")
    if isinstance(name, dict):
        return str(name.get("full") or name.get("userPreferred") or name.get("native") or "")
    return str(name or "")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = read_json(args.tags_input)
    safe_tags = load_safe_llm_personality(args.safe_llm_tags)

    raw_assignments = []
    descriptor_counts: Counter[str] = Counter()
    descriptor_character_ids: dict[str, set[int]] = defaultdict(set)
    descriptor_sources: dict[str, Counter[str]] = defaultdict(Counter)
    descriptor_examples: dict[str, list[dict]] = defaultdict(list)

    for source in payload.get("characters", []):
        character_id = int(source.get("anilist_character_id") or 0)
        if not character_id:
            continue
        rows = descriptor_rows_from_character(source, safe_tags)
        seen_for_character = set()
        for row in rows:
            descriptor = row.get("tag") or ""
            if not descriptor:
                continue
            raw_assignments.append(
                {
                    "anilist_character_id": character_id,
                    "name": character_name(source),
                    "first_anime": source.get("first_anime"),
                    "descriptor": descriptor,
                    "source": row.get("source"),
                    "evidence": row.get("evidence"),
                }
            )
            descriptor_counts[descriptor] += 1
            descriptor_sources[descriptor][str(row.get("source") or "unknown")] += 1
            if descriptor not in seen_for_character:
                descriptor_character_ids[descriptor].add(character_id)
                seen_for_character.add(descriptor)
            if len(descriptor_examples[descriptor]) < args.max_examples_per_descriptor:
                descriptor_examples[descriptor].append(raw_assignments[-1])

    raw_descriptors = sorted(descriptor_counts)
    embeddings = load_or_create_embeddings(raw_descriptors, args.output_dir, args.embedding_model)
    canonical_descriptors, canonical_embeddings, raw_to_canonical, canonical_groups = canonicalize_descriptors(
        raw_descriptors,
        embeddings,
        args.similarity_threshold,
        args.contained_distance_threshold,
    )
    canonical_index = {descriptor: index for index, descriptor in enumerate(canonical_descriptors)}

    by_canonical: dict[str, dict] = {}
    for raw_descriptor in raw_descriptors:
        canonical = raw_to_canonical[raw_descriptor]
        row = by_canonical.setdefault(
            canonical,
            {
                "canonical": canonical,
                "members": [],
                "assignment_count": 0,
                "character_ids": set(),
                "sources": Counter(),
                "examples": [],
            },
        )
        row["members"].append(raw_descriptor)
        row["assignment_count"] += descriptor_counts[raw_descriptor]
        row["character_ids"].update(descriptor_character_ids[raw_descriptor])
        row["sources"].update(descriptor_sources[raw_descriptor])
        for example in descriptor_examples[raw_descriptor]:
            if len(row["examples"]) < args.max_examples_per_descriptor:
                row["examples"].append(example)

    descriptors = []
    for row in by_canonical.values():
        character_ids = row.pop("character_ids")
        row["members"] = sorted(row["members"], key=lambda value: (len(descriptor_tokens(value)), len(value), value))
        row["member_count"] = len(row["members"])
        row["character_count"] = len(character_ids)
        row["sources"] = dict(row["sources"])
        descriptors.append(row)
    descriptors.sort(key=lambda row: (-row["character_count"], -row["assignment_count"], row["canonical"]))

    output = {
        "generated_at": utc_now(),
        "source": "cache_global_descriptor_canonicalization.py",
        "parameters": {
            "tags_input": str(args.tags_input),
            "safe_llm_tags": [str(path) for path in args.safe_llm_tags],
            "embedding_model": args.embedding_model,
            "similarity_threshold": args.similarity_threshold,
            "contained_distance_threshold": args.contained_distance_threshold,
            "canonical_representative": "fewest token words, then shortest character length, then alphabetical",
            "contained_merge_rule": "longer descriptor can merge into shorter token-subphrase when cosine distance is below threshold",
        },
        "counts": {
            "characters": len({row["anilist_character_id"] for row in raw_assignments}),
            "raw_assignments": len(raw_assignments),
            "raw_descriptors": len(raw_descriptors),
            "canonical_descriptors": len(canonical_descriptors),
            "canonical_merge_groups": len(canonical_groups),
        },
        "raw_to_canonical": raw_to_canonical,
        "canonical_descriptor_groups": canonical_groups,
        "descriptors": descriptors,
    }

    output_path = args.output_dir / "descriptor_canonicalization.json"
    write_json(output_path, output)
    np.savez_compressed(
        args.output_dir / "descriptor_canonicalization_embeddings.npz",
        descriptors=np.asarray(canonical_descriptors, dtype=object),
        embeddings=canonical_embeddings.astype(np.float32),
    )
    print(json.dumps(output["counts"], indent=2))
    print(f"wrote {output_path}")
    print(f"wrote {args.output_dir / 'descriptor_canonicalization_embeddings.npz'}")


if __name__ == "__main__":
    main()
