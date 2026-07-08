#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


POSITIVE_ANCHORS = [
    "personality trait",
    "temperament",
    "behavioral archetype",
    "recurring emotional pattern",
    "social attitude",
    "disposition",
    "personality adjective",
    "emotional trait",
    "romantic attitude",
    "romantic social behavior",
    "social charm",
    "easygoing temperament",
    "relaxed personality",
    "easygoing laid-back temperament",
]

NEGATIVE_ANCHORS = [
    "nationality",
    "occupation",
    "species",
    "physical appearance",
    "combat ability",
    "story role",
    "relationship label",
    "setting",
    "age group",
    "temporary emotional state",
    "momentary reaction",
    "emotional episode",
    "transient mood",
    "reaction to event",
    "facial expression",
    "embarrassed reaction",
    "learned skill",
    "technical skill",
    "language ability",
    "physical weakness",
    "medical condition",
    "physical condition",
    "academic ability",
    "personal pronoun or self-reference",
    "first-person pronoun",
    "speech register or verbal tic",
    "grammatical pronoun",
    "domain adjective requiring a head noun",
    "relational domain modifier",
    "interpersonal domain label",
    "communication-domain label",
    "abstract category label",
    "time frequency or schedule",
    "daily routine or ordinary activity",
    "temporal adjective",
    "physical disability or impairment",
    "bodily limitation",
    "injury or disability status",
    "health limitation or medical impairment",
    "disabled physical state",
    "incomplete relational adjective",
    "requires an object or complement",
    "directional relation or preference",
    "orientation toward another object",
    "fragment of a phrasal adjective",
    "participle state fragment",
    "past tense action fragment",
    "physical posture or position",
    "result of an action",
    "lying down physical position",
    "passive participle of action verb",
    "orientation or alignment word",
    "political or directional orientation",
    "bare suffix of a compound adjective",
    "compound adjective tail fragment",
    "generic tail word requiring a compound modifier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score canonical personality descriptors against positive/negative semantic anchors using "
            "the descriptor evidence context, not the bare descriptor string."
        )
    )
    parser.add_argument(
        "--descriptor-assignments",
        type=Path,
        default=Path("run/adjectival_personality_union/adjectival_personality_assignments.jsonl"),
    )
    parser.add_argument(
        "--global-canonicalization-input",
        type=Path,
        default=Path("models/global_descriptor_canonicalization/descriptor_canonicalization.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/contextual_personality_anchor_scores/contextual_personality_anchor_scores.json"),
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-score", type=float, default=0.005)
    parser.add_argument("--examples-per-descriptor", type=int, default=5)
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


def context_text(row: dict[str, Any]) -> str:
    tag = str(row.get("tag") or row.get("original_tag") or "").strip()
    evidence = str(row.get("evidence") or "").strip()
    return f"descriptor: {tag}. evidence: {evidence}"


def load_rows(assignments_path: Path, raw_to_canonical: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    rows_by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with assignments_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = str(row.get("tag") or "").strip()
            if not raw:
                continue
            canonical = raw_to_canonical.get(raw, raw)
            rows_by_canonical[canonical].append(row)
    return dict(rows_by_canonical)


def main() -> None:
    args = parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install sentence-transformers to score contextual descriptor anchors.") from error

    canonicalization = read_json(args.global_canonicalization_input) if args.global_canonicalization_input.exists() else {}
    raw_to_canonical = {
        str(raw): str(canonical)
        for raw, canonical in (canonicalization.get("raw_to_canonical") or {}).items()
    }
    rows_by_canonical = load_rows(args.descriptor_assignments, raw_to_canonical)

    descriptors = sorted(rows_by_canonical)
    texts: list[str] = []
    text_to_descriptor: list[str] = []
    text_to_row: list[dict[str, Any]] = []
    for descriptor in descriptors:
        for row in rows_by_canonical[descriptor]:
            texts.append(context_text(row))
            text_to_descriptor.append(descriptor)
            text_to_row.append(row)

    model = SentenceTransformer(args.embedding_model)
    anchor_texts = POSITIVE_ANCHORS + NEGATIVE_ANCHORS
    anchor_embeddings = model.encode(
        anchor_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float64)
    positive_embeddings = anchor_embeddings[: len(POSITIVE_ANCHORS)]
    negative_embeddings = anchor_embeddings[len(POSITIVE_ANCHORS) :]

    descriptor_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        embeddings = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float64)
        positive_scores = embeddings @ positive_embeddings.T
        negative_scores = embeddings @ negative_embeddings.T
        for offset, embedding in enumerate(embeddings):
            row_index = start + offset
            positive_index = int(np.argmax(positive_scores[offset]))
            negative_index = int(np.argmax(negative_scores[offset]))
            positive_score = float(positive_scores[offset, positive_index])
            negative_score = float(negative_scores[offset, negative_index])
            descriptor = text_to_descriptor[row_index]
            source_row = text_to_row[row_index]
            descriptor_scores[descriptor].append(
                {
                    "personality_score": positive_score - negative_score,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "positive_anchor": POSITIVE_ANCHORS[positive_index],
                    "negative_anchor": NEGATIVE_ANCHORS[negative_index],
                    "character_id": source_row.get("anilist_character_id"),
                    "name": source_row.get("name"),
                    "first_anime": source_row.get("first_anime"),
                    "tag": source_row.get("tag"),
                    "evidence": source_row.get("evidence"),
                    "source": source_row.get("source"),
                }
            )

    descriptor_rows = []
    for descriptor in descriptors:
        scored = descriptor_scores[descriptor]
        scores = np.asarray([row["personality_score"] for row in scored], dtype=np.float64)
        positive = np.asarray([row["positive_score"] for row in scored], dtype=np.float64)
        negative = np.asarray([row["negative_score"] for row in scored], dtype=np.float64)
        examples = sorted(scored, key=lambda row: row["personality_score"], reverse=True)[: args.examples_per_descriptor]
        hard_examples = sorted(scored, key=lambda row: row["personality_score"])[: args.examples_per_descriptor]
        descriptor_rows.append(
            {
                "descriptor": descriptor,
                "assignment_count": len(scored),
                "character_count": len({row.get("character_id") for row in scored}),
                "personality_score_mean": round(float(np.mean(scores)), 8),
                "personality_score_median": round(float(np.median(scores)), 8),
                "personality_score_min": round(float(np.min(scores)), 8),
                "personality_score_max": round(float(np.max(scores)), 8),
                "positive_score_mean": round(float(np.mean(positive)), 8),
                "negative_score_mean": round(float(np.mean(negative)), 8),
                "keep_default": bool(float(np.mean(scores)) >= args.min_score),
                "examples": examples,
                "hard_examples": hard_examples,
            }
        )
    descriptor_rows.sort(
        key=lambda row: (-row["personality_score_mean"], -row["character_count"], row["descriptor"])
    )

    payload = {
        "generated_at": utc_now(),
        "source": "cache_contextual_personality_anchor_scores.py",
        "parameters": {
            "descriptor_assignments": str(args.descriptor_assignments),
            "global_canonicalization_input": str(args.global_canonicalization_input),
            "embedding_model": args.embedding_model,
            "min_score": args.min_score,
            "positive_anchors": POSITIVE_ANCHORS,
            "negative_anchors": NEGATIVE_ANCHORS,
            "context_template": "descriptor: <tag>. evidence: <evidence>",
            "score": "max cosine to positive anchors minus max cosine to negative anchors, computed on evidence-context text",
        },
        "counts": {
            "descriptors": len(descriptor_rows),
            "assignments": len(texts),
            "kept_default": sum(1 for row in descriptor_rows if row["keep_default"]),
        },
        "descriptors": descriptor_rows,
        "scores_by_descriptor": {row["descriptor"]: row for row in descriptor_rows},
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
