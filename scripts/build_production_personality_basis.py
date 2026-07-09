#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_adjectival_personality_union import adjectival_canonical, normalize_tag  # noqa: E402
from cache_contextual_personality_anchor_scores import NEGATIVE_ANCHORS, POSITIVE_ANCHORS  # noqa: E402
from seiyuu_character_semantic_clusters import canonicalize_descriptors  # noqa: E402
from seiyuu_local_nmf_lane_svd import load_or_create_embeddings, read_json, utc_now, write_json  # noqa: E402


DEFAULT_ANILIST = Path("data/external/llm/all_character_description_tags_canonical.json")
DEFAULT_VNDB_MERGED = Path("data/external/merged/all_characters_llm_vndb_personality_tags.json")
DEFAULT_BANGUMI = [
    Path("data/external/safe_enrichment_llm/character_tags.jsonl"),
    Path(
        "run/gpu_llm_tagging/a100_batch_transformers_prod_20260701/"
        "batch_transformers_prod_complete/character_tags_deduped_aggressive.jsonl"
    ),
    Path("run/gpu_llm_tagging/returned_latest/character_tags.jsonl"),
]
DEFAULT_COMMON_ADJECTIVES = Path(
    "models/simple_adjective_basis/simple_common_adjectives_en_c5000_r384_centernone_baai_bge-small-en-v1.5.json"
)
DEFAULT_GLOBAL_CANONICALIZATION = Path("models/global_descriptor_canonicalization/descriptor_canonicalization.json")
DEFAULT_CURATION = Path("config/personality_descriptor_curation.json")
DEFAULT_LLM_DECISIONS = Path("run/production_personality_basis/llm_personality_decisions_v5.jsonl")

HARD_NEGATIVE_ANCHORS = {
    "time frequency or schedule",
    "daily routine or ordinary activity",
    "temporal adjective",
    "physical disability or impairment",
    "bodily limitation",
    "injury or disability status",
    "health limitation or medical impairment",
    "disabled physical state",
    "personal pronoun or self-reference",
    "first-person pronoun",
    "speech register or verbal tic",
    "grammatical pronoun",
    "domain adjective requiring a head noun",
    "relational domain modifier",
    "interpersonal domain label",
    "communication-domain label",
    "abstract category label",
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
}

MARGIN_HARD_NEGATIVE_ANCHORS = {
    "domain adjective requiring a head noun",
    "relational domain modifier",
    "interpersonal domain label",
    "communication-domain label",
    "abstract category label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a production personality-descriptor basis from common WordNet adjectives plus all "
            "AniList/Bangumi LLM tags."
        )
    )
    parser.add_argument("--anilist-tags", type=Path, default=DEFAULT_ANILIST)
    parser.add_argument(
        "--vndb-merged-tags",
        type=Path,
        default=DEFAULT_VNDB_MERGED,
        help=(
            "Merged AniList/VNDB character tag file. Only VNDB-origin non-role tags are read from this file; "
            "AniList LLM tags still come from --anilist-tags to avoid double counting."
        ),
    )
    parser.add_argument("--bangumi-tags", nargs="*", type=Path, default=DEFAULT_BANGUMI)
    parser.add_argument("--common-adjectives", type=Path, default=DEFAULT_COMMON_ADJECTIVES)
    parser.add_argument(
        "--global-canonicalization",
        type=Path,
        default=DEFAULT_GLOBAL_CANONICALIZATION,
        help="Existing raw-to-canonical descriptor map applied before the compact adjective gate.",
    )
    parser.add_argument(
        "--curation",
        type=Path,
        default=DEFAULT_CURATION,
        help="Descriptor rejection lockouts. Used to preserve accepted curation decisions across rebuilds.",
    )
    parser.add_argument(
        "--llm-decisions",
        type=Path,
        default=DEFAULT_LLM_DECISIONS,
        help=(
            "Cached deterministic LLM descriptor decisions. When present, this is the semantic keep/reject "
            "gate; anchor hard negatives remain broad non-personality overrides."
        ),
    )
    parser.add_argument("--common-adjective-count", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("run/production_personality_basis"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-words", type=int, default=4)
    parser.add_argument(
        "--single-word-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep compact adjective atoms only; hyphenated words are allowed.",
    )
    parser.add_argument(
        "--canonicalize-similarity-threshold",
        type=float,
        default=1.01,
        help="Embedding-similarity merge threshold; >1 disables synonym merging.",
    )
    parser.add_argument(
        "--canonicalize-contained-distance-threshold",
        type=float,
        default=0.16,
        help="Merge longer descriptors into contained shorter descriptors when cosine distance is small.",
    )
    parser.add_argument("--min-personality-score", type=float, default=0.005)
    parser.add_argument("--min-character-count", type=int, default=0)
    parser.add_argument("--examples-per-descriptor", type=int, default=3)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compact_shape(value: str) -> bool:
    return re.fullmatch(r"[a-z]+(?:-[a-z]+)*", value or "") is not None


def hyphen_bundle_fragment(value: str) -> bool:
    """Reject glued lists like a-b-c; keep ordinary two-part compounds."""
    return len([part for part in normalize_tag(value).split("-") if part]) >= 3


def load_curation_rejections(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = read_json(path)
    raw_rejections = payload.get("reject_descriptors") or {}
    if isinstance(raw_rejections, list):
        return {normalize_tag(str(descriptor)): "curated reject" for descriptor in raw_rejections}
    return {
        normalize_tag(str(descriptor)): str(reason or "curated reject")
        for descriptor, reason in raw_rejections.items()
        if normalize_tag(str(descriptor))
    }


def curated_keep(base_keep: bool, descriptor: str, rejection_reasons: dict[str, str]) -> tuple[bool, str]:
    reason = rejection_reasons.get(normalize_tag(descriptor))
    if reason:
        return False, reason
    return bool(base_keep), ""


def load_llm_decisions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            descriptor = normalize_tag(str(row.get("descriptor") or ""))
            if not descriptor:
                continue
            decisions[descriptor] = {
                "llm_keep": bool(row.get("llm_keep")),
                "llm_reason": str(row.get("llm_reason") or ""),
                "llm_model": str(row.get("llm_model") or ""),
            }
    return decisions


def anchor_hard_reject(best_negative_anchor: str, score_mean: float) -> bool:
    if best_negative_anchor in MARGIN_HARD_NEGATIVE_ANCHORS:
        return score_mean < -0.05
    return best_negative_anchor in HARD_NEGATIVE_ANCHORS and score_mean < 0.0


def compound_tail_fragments(descriptors: list[str]) -> set[str]:
    descriptor_set = set(descriptors)
    tails: set[str] = set()
    for descriptor in descriptor_set:
        if "-" not in descriptor:
            continue
        tail = descriptor.rsplit("-", 1)[-1]
        if tail in descriptor_set:
            tails.add(tail)
    return tails


def semantic_keep(
    descriptor: str,
    anchor_keep: bool,
    best_negative_anchor: str,
    score_mean: float,
    llm_decisions: dict[str, dict[str, Any]],
    tail_fragments: set[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    if hyphen_bundle_fragment(descriptor):
        return False, {
            "anchor_keep": anchor_keep,
            "llm_keep": llm_decisions.get(normalize_tag(descriptor), {}).get("llm_keep", ""),
            "llm_reason": llm_decisions.get(normalize_tag(descriptor), {}).get("llm_reason", ""),
            "llm_model": llm_decisions.get(normalize_tag(descriptor), {}).get("llm_model", ""),
            "anchor_hard_reject": True,
            "decision_reason": "hyphen_bundle_fragment",
        }
    if normalize_tag(descriptor) in (tail_fragments or set()):
        return False, {
            "anchor_keep": anchor_keep,
            "llm_keep": llm_decisions.get(normalize_tag(descriptor), {}).get("llm_keep", ""),
            "llm_reason": llm_decisions.get(normalize_tag(descriptor), {}).get("llm_reason", ""),
            "llm_model": llm_decisions.get(normalize_tag(descriptor), {}).get("llm_model", ""),
            "anchor_hard_reject": True,
            "decision_reason": "compound_tail_fragment",
        }
    decision = llm_decisions.get(normalize_tag(descriptor))
    hard_reject = anchor_hard_reject(best_negative_anchor, score_mean)
    if decision is not None:
        keep = bool(decision["llm_keep"]) and not hard_reject
        reason = "llm_decision"
        if hard_reject:
            reason = "hard_negative_anchor"
        elif not decision["llm_keep"]:
            reason = "llm_reject"
        return keep, {
            "anchor_keep": anchor_keep,
            "llm_keep": decision["llm_keep"],
            "llm_reason": decision["llm_reason"],
            "llm_model": decision["llm_model"],
            "anchor_hard_reject": hard_reject,
            "decision_reason": reason,
        }
    keep = bool(anchor_keep) and not hard_reject
    return keep, {
        "anchor_keep": anchor_keep,
        "llm_keep": "",
        "llm_reason": "",
        "llm_model": "",
        "anchor_hard_reject": hard_reject,
        "decision_reason": "anchor_score" if not hard_reject else "hard_negative_anchor",
    }


def iter_common_adjectives(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = []
    for rank, row in enumerate((payload.get("candidates") or [])[:limit], start=1):
        descriptor = normalize_tag(str(row.get("descriptor") or ""))
        if not descriptor:
            continue
        rows.append(
            {
                "candidate": descriptor,
                "source": "wordfreq_wordnet_common_adjective",
                "category": "common_adjective",
                "character_id": None,
                "name": None,
                "first_anime": None,
                "evidence": f"common WordNet adjective rank {rank}",
                "confidence": None,
                "frequency_rank": int(row.get("frequency_rank") or rank),
                "zipf_frequency": row.get("zipf_frequency"),
            }
        )
    return rows


def iter_anilist(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = []
    for character in payload.get("characters", []):
        tags = character.get("llm_tags") or {}
        for category in ("role", "personality", "traits"):
            for tag in tags.get(category) or []:
                rows.append(
                    {
                        "candidate": tag.get("tag"),
                        "source": "anilist_description_qwen",
                        "category": category,
                        "character_id": character.get("anilist_character_id"),
                        "name": character.get("name"),
                        "first_anime": character.get("first_anime"),
                        "evidence": tag.get("evidence"),
                        "confidence": tag.get("confidence"),
                        "frequency_rank": None,
                        "zipf_frequency": None,
                    }
                )
    return rows


def iter_vndb(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    rows = []
    for character in payload.get("characters", []):
        seen: set[tuple[str, str]] = set()
        tags = character.get("llm_tags") or {}
        for category in ("personality", "traits"):
            for tag in tags.get(category) or []:
                sources = {str(source).lower() for source in (tag.get("sources") or [])}
                if "vndb" not in sources:
                    continue
                groups = [str(group) for group in (tag.get("vndb_groups") or [])]
                if any(group.lower() == "role" for group in groups):
                    continue
                value = tag.get("tag")
                key = (category, str(value or "").strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "candidate": value,
                        "source": "vndb_personality_traits",
                        "category": category,
                        "character_id": character.get("anilist_character_id"),
                        "name": character.get("name"),
                        "first_anime": character.get("first_anime"),
                        "evidence": "VNDB character trait"
                        + (f" ({', '.join(groups)})" if groups else ""),
                        "confidence": tag.get("confidence") or "vndb",
                        "vndb_groups": groups,
                        "vndb_char_count": tag.get("vndb_char_count"),
                        "frequency_rank": None,
                        "zipf_frequency": None,
                    }
                )
    return rows


def iter_jsonl_tags(path: Path, source_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            character = json.loads(line)
            tags = character.get("tags") or {}
            for category in ("role", "personality", "traits"):
                for tag in tags.get(category) or []:
                    rows.append(
                        {
                            "candidate": tag.get("tag"),
                            "source": source_name,
                            "category": category,
                            "character_id": character.get("anilist_character_id"),
                            "name": character.get("name"),
                            "first_anime": character.get("first_anime"),
                            "evidence": tag.get("evidence"),
                            "confidence": tag.get("confidence"),
                            "source_key": tag.get("source_key"),
                            "source_url": tag.get("source_url"),
                            "frequency_rank": None,
                            "zipf_frequency": None,
                        }
                    )
    return rows


def normalize_candidate(
    row: dict[str, Any],
    max_words: int,
    single_word_only: bool,
    raw_to_canonical: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    raw_to_canonical = raw_to_canonical or {}
    raw_candidate = normalize_tag(str(row.get("candidate") or ""))
    mapped_candidate = raw_to_canonical.get(raw_candidate, raw_candidate)
    canonical, reason = adjectival_canonical(mapped_candidate, max_words)
    if not canonical:
        return None, reason
    canonical = canonical.replace("_", "-")
    if single_word_only and not compact_shape(canonical):
        compact_mapped = mapped_candidate.replace(" ", "-")
        if compact_shape(compact_mapped) and canonical.replace(" ", "-") == compact_mapped:
            canonical = compact_mapped
    if single_word_only and not compact_shape(canonical):
        return None, "not_single_word_or_hyphenated"
    return canonical, reason


def descriptor_context(row: dict[str, Any]) -> str:
    descriptor = str(row.get("canonical") or row.get("normalized") or row.get("candidate") or "").strip()
    evidence = str(row.get("evidence") or "").strip()
    if row.get("source") == "wordfreq_wordnet_common_adjective":
        evidence = f"a fictional character is described as {descriptor}"
    return f"descriptor: {descriptor}. evidence: {evidence}"


def score_contexts(texts: list[str], model_name: str, batch_size: int) -> list[dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    anchor_texts = POSITIVE_ANCHORS + NEGATIVE_ANCHORS
    anchor_embeddings = model.encode(
        anchor_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float64)
    pos = anchor_embeddings[: len(POSITIVE_ANCHORS)]
    neg = anchor_embeddings[len(POSITIVE_ANCHORS) :]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        embeddings = model.encode(
            texts[start : start + batch_size],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float64)
        pos_scores = embeddings @ pos.T
        neg_scores = embeddings @ neg.T
        for offset in range(embeddings.shape[0]):
            pos_index = int(np.argmax(pos_scores[offset]))
            neg_index = int(np.argmax(neg_scores[offset]))
            pos_score = float(pos_scores[offset, pos_index])
            neg_score = float(neg_scores[offset, neg_index])
            rows.append(
                {
                    "personality_score": pos_score - neg_score,
                    "positive_score": pos_score,
                    "negative_score": neg_score,
                    "positive_anchor": POSITIVE_ANCHORS[pos_index],
                    "negative_anchor": NEGATIVE_ANCHORS[neg_index],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curation_rejections = load_curation_rejections(args.curation)
    llm_decisions = load_llm_decisions(args.llm_decisions)

    raw_rows = []
    raw_rows.extend(iter_common_adjectives(args.common_adjectives, args.common_adjective_count))
    raw_rows.extend(iter_anilist(args.anilist_tags))
    raw_rows.extend(iter_vndb(args.vndb_merged_tags))
    for path in args.bangumi_tags:
        raw_rows.extend(iter_jsonl_tags(path, f"bangumi_qwen:{path.name}"))

    global_canonicalization = read_json(args.global_canonicalization) if args.global_canonicalization.exists() else {}
    raw_to_canonical = {
        normalize_tag(str(raw)): normalize_tag(str(canonical))
        for raw, canonical in (global_canonicalization.get("raw_to_canonical") or {}).items()
    }

    normalized_rows: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    for row in raw_rows:
        normalized, reason = normalize_candidate(row, args.max_words, args.single_word_only, raw_to_canonical)
        if not normalized:
            reject_reasons[reason] += 1
            continue
        normalized_rows.append({**row, "normalized": normalized, "adjectival_mapping": reason})

    # Exact lexical dedupe before embedding canonicalization.
    descriptors = sorted({row["normalized"] for row in normalized_rows})
    descriptor_embeddings = load_or_create_embeddings(descriptors, args.output_dir, args.embedding_model)
    descriptor_embeddings = descriptor_embeddings / np.maximum(
        np.linalg.norm(descriptor_embeddings, axis=1, keepdims=True),
        1.0e-12,
    )
    canonical_descriptors, canonical_embeddings, raw_to_canonical, canonical_groups = canonicalize_descriptors(
        descriptors,
        descriptor_embeddings,
        args.canonicalize_similarity_threshold,
        args.canonicalize_contained_distance_threshold,
    )
    canonical_set = set(canonical_descriptors)

    rows_by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        canonical = raw_to_canonical.get(row["normalized"], row["normalized"])
        if canonical not in canonical_set:
            continue
        rows_by_canonical[canonical].append({**row, "canonical": canonical})

    context_rows: list[tuple[str, dict[str, Any]]] = []
    for descriptor in sorted(rows_by_canonical):
        # Score all LLM evidence rows, but only one neutral row for common-adjective-only descriptors.
        llm_rows = [
            row
            for row in rows_by_canonical[descriptor]
            if row.get("source") != "wordfreq_wordnet_common_adjective"
        ]
        if llm_rows:
            for row in llm_rows:
                context_rows.append((descriptor, row))
        else:
            context_rows.append((descriptor, rows_by_canonical[descriptor][0]))

    scored_contexts = score_contexts(
        [descriptor_context(row) for _, row in context_rows],
        args.embedding_model,
        args.batch_size,
    )
    score_rows_by_descriptor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (descriptor, source_row), score_row in zip(context_rows, scored_contexts, strict=True):
        score_rows_by_descriptor[descriptor].append({**source_row, **score_row})

    output_rows = []
    tail_fragments = compound_tail_fragments(sorted(rows_by_canonical))
    for descriptor in sorted(rows_by_canonical):
        all_rows = rows_by_canonical[descriptor]
        scored = score_rows_by_descriptor[descriptor]
        scores = np.asarray([row["personality_score"] for row in scored], dtype=np.float64)
        positive = np.asarray([row["positive_score"] for row in scored], dtype=np.float64)
        negative = np.asarray([row["negative_score"] for row in scored], dtype=np.float64)
        source_counts = Counter(str(row.get("source") or "unknown") for row in all_rows)
        category_counts = Counter(str(row.get("category") or "unknown") for row in all_rows)
        characters = {int(row["character_id"]) for row in all_rows if row.get("character_id") is not None}
        common_rows = [row for row in all_rows if row.get("source") == "wordfreq_wordnet_common_adjective"]
        examples = sorted(scored, key=lambda row: row["personality_score"], reverse=True)[: args.examples_per_descriptor]
        score_mean = float(np.mean(scores))
        best_negative_anchor = examples[0]["negative_anchor"] if examples else ""
        anchor_keep = bool(score_mean >= args.min_personality_score and len(characters) >= args.min_character_count)
        base_keep, decision_details = semantic_keep(
            descriptor,
            anchor_keep,
            best_negative_anchor,
            score_mean,
            llm_decisions,
            tail_fragments,
        )
        keep, curation_reason = curated_keep(base_keep, descriptor, curation_rejections)
        if curation_reason:
            decision_details = {**decision_details, "decision_reason": "curated_reject"}
        output_rows.append(
            {
                "descriptor": descriptor,
                "keep": keep,
                "curation_reason": curation_reason,
                "decision_reason": decision_details["decision_reason"],
                "anchor_keep": decision_details["anchor_keep"],
                "llm_keep": decision_details["llm_keep"],
                "anchor_hard_reject": decision_details["anchor_hard_reject"],
                "llm_model": decision_details["llm_model"],
                "llm_reason": decision_details["llm_reason"],
                "personality_score_mean": round(score_mean, 8),
                "personality_score_median": round(float(np.median(scores)), 8),
                "personality_score_min": round(float(np.min(scores)), 8),
                "personality_score_max": round(float(np.max(scores)), 8),
                "positive_score_mean": round(float(np.mean(positive)), 8),
                "negative_score_mean": round(float(np.mean(negative)), 8),
                "best_positive_anchor": examples[0]["positive_anchor"] if examples else "",
                "best_negative_anchor": best_negative_anchor,
                "assignment_count": len(all_rows),
                "scored_context_count": len(scored),
                "character_count": len(characters),
                "source_counts": json.dumps(dict(sorted(source_counts.items())), ensure_ascii=False),
                "category_counts": json.dumps(dict(sorted(category_counts.items())), ensure_ascii=False),
                "wordfreq_rank": min((int(row.get("frequency_rank") or 10**9) for row in common_rows), default=""),
                "zipf_frequency": max((float(row.get("zipf_frequency") or 0.0) for row in common_rows), default=""),
                "examples": json.dumps(
                    [
                        {
                            "source": row.get("source"),
                            "category": row.get("category"),
                            "character_id": row.get("character_id"),
                            "name": row.get("name"),
                            "first_anime": row.get("first_anime"),
                            "evidence": row.get("evidence"),
                            "personality_score": round(float(row["personality_score"]), 6),
                            "positive_anchor": row.get("positive_anchor"),
                            "negative_anchor": row.get("negative_anchor"),
                        }
                        for row in examples
                    ],
                    ensure_ascii=False,
                ),
            }
        )

    output_rows.sort(key=lambda row: row["descriptor"])
    kept_rows = [row for row in output_rows if row["keep"]]

    tsv_path = args.output_dir / "production_personality_basis.tsv"
    csv_path = args.output_dir / "production_personality_basis.csv"
    kept_tsv_path = args.output_dir / "production_personality_basis_kept.tsv"
    kept_csv_path = args.output_dir / "production_personality_basis_kept.csv"
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    for path, dialect, rows in [
        (tsv_path, "excel-tab", output_rows),
        (csv_path, "excel", output_rows),
        (kept_tsv_path, "excel-tab", kept_rows),
        (kept_csv_path, "excel", kept_rows),
    ]:
        with path.with_name(f"{path.name}.tmp").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, dialect=dialect)
            writer.writeheader()
            writer.writerows(rows)
        path.with_name(f"{path.name}.tmp").replace(path)

    write_json(
        args.output_dir / "production_personality_basis_summary.json",
        {
            "generated_at": now(),
            "source": "build_production_personality_basis.py",
            "parameters": {
                "anilist_tags": str(args.anilist_tags),
                "vndb_merged_tags": str(args.vndb_merged_tags),
                "bangumi_tags": [str(path) for path in args.bangumi_tags],
                "common_adjectives": str(args.common_adjectives),
                "global_canonicalization": str(args.global_canonicalization),
                "curation": str(args.curation),
                "llm_decisions": str(args.llm_decisions),
                "common_adjective_count": args.common_adjective_count,
                "embedding_model": args.embedding_model,
                "max_words": args.max_words,
                "single_word_only": args.single_word_only,
                "canonicalize_similarity_threshold": args.canonicalize_similarity_threshold,
                "canonicalize_contained_distance_threshold": args.canonicalize_contained_distance_threshold,
                "min_personality_score": args.min_personality_score,
                "min_character_count": args.min_character_count,
                "curated_reject_count": len(curation_rejections),
                "llm_decision_count": len(llm_decisions),
                "positive_anchors": POSITIVE_ANCHORS,
                "negative_anchors": NEGATIVE_ANCHORS,
                "hard_negative_anchors": sorted(HARD_NEGATIVE_ANCHORS),
                "decider": (
                    "cached deterministic LLM descriptor decision when available; otherwise contextual BGE "
                    "anchor score; broad hard-negative anchors override both"
                ),
            },
            "counts": {
                "raw_candidates": len(raw_rows),
                "normalized_candidates": len(normalized_rows),
                "exact_descriptor_count": len(descriptors),
                "canonical_descriptor_count": len(canonical_descriptors),
                "canonical_merge_groups": len(canonical_groups),
                "final_rows": len(output_rows),
                "kept_rows": len(kept_rows),
                "reject_reasons": dict(reject_reasons),
            },
            "outputs": {
                "tsv": str(tsv_path),
                "csv": str(csv_path),
                "kept_tsv": str(kept_tsv_path),
                "kept_csv": str(kept_csv_path),
            },
            "top_kept": kept_rows[:25],
        },
    )
    print(
        json.dumps(
            {
                "raw_candidates": len(raw_rows),
                "normalized_candidates": len(normalized_rows),
                "canonical_descriptor_count": len(canonical_descriptors),
                "kept_rows": len(kept_rows),
                "tsv": str(tsv_path),
                "csv": str(csv_path),
                "kept_tsv": str(kept_tsv_path),
                "kept_csv": str(kept_csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
