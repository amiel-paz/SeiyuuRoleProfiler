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

import nltk
from nltk.corpus import wordnet as wn


DEFAULT_SOURCES = {
    "anilist_description_qwen": Path("data/external/llm/all_character_description_tags_canonical.json"),
    "vndb_personality_traits": Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"),
    "bangumi_local_ollama_qwen": Path("data/external/safe_enrichment_llm/character_tags.jsonl"),
    "bangumi_a100_batch_qwen": Path(
        "run/gpu_llm_tagging/a100_batch_transformers_prod_20260701/"
        "batch_transformers_prod_complete/character_tags_deduped_aggressive.jsonl"
    ),
}
DEFAULT_GLOBAL_CANONICALIZATION = Path("models/global_descriptor_canonicalization/descriptor_canonicalization.json")

FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "can",
    "for",
    "from",
    "in",
    "into",
    "not",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
}

GENERIC_NON_PERSONALITY = {
    "attitude",
    "behavior",
    "behavior style",
    "interpersonal style",
    "personality",
    "stable temperament",
    "temperament",
}

LEGAL_OR_DOMAIN_STATUS_WORDS = {
    "illegal",
}

DOMAIN_ONLY_DEFINITION_MARKERS = {
    "being everywhere equidistant",
    "direction or description in music",
    "having or showing knowledge and skill and aptitude",
    "of or relating to behavior",
    "of or relating to the simultaneous performance",
    "prohibited by law",
    "skillful after much practice",
}

TEMPORARY_STATE_MARKERS = {
    "compulsively or physiologically dependent",
    "experiencing",
    "feeling",
    "frenzied as if possessed",
    "stricken",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an adjective-clean union of personality tags from all LLM sources.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("run/adjectival_personality_union/adjectival_personality_union.json"),
    )
    parser.add_argument(
        "--assignments-output",
        type=Path,
        default=Path("run/adjectival_personality_union/adjectival_personality_assignments.jsonl"),
    )
    parser.add_argument(
        "--global-canonicalization",
        type=Path,
        default=DEFAULT_GLOBAL_CANONICALIZATION,
        help="Existing raw-to-canonical descriptor map applied before the adjectival gate.",
    )
    parser.add_argument("--max-words", type=int, default=4)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def load_raw_to_canonical(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        normalize_tag(str(raw)): normalize_tag(str(canonical))
        for raw, canonical in (payload.get("raw_to_canonical") or {}).items()
    }


def tokens_for(value: str) -> list[str]:
    return [token for token in re.split(r"[\s-]+", value) if token]


def has_adjective_synset(value: str) -> bool:
    return bool(wn.synsets(value, pos=wn.ADJ) or wn.synsets(value, pos=wn.ADJ_SAT))


def has_nonadjective_synset(value: str) -> bool:
    return bool(wn.synsets(value, pos=wn.NOUN) or wn.synsets(value, pos=wn.VERB) or wn.synsets(value, pos=wn.ADV))


def candidate_forms(value: str) -> list[str]:
    tokens = tokens_for(value)
    if not tokens:
        return []
    spaced = " ".join(tokens)
    hyphenated = "-".join(tokens)
    underscored = "_".join(tokens)
    joined = "".join(tokens)
    return list(dict.fromkeys([spaced, hyphenated, underscored, joined]))


def full_phrase_adjective(value: str) -> str | None:
    for form in candidate_forms(value):
        if has_adjective_synset(form):
            return form.replace("_", "-")
    return None


def pos_in_adjective_context(tokens: list[str]) -> list[str]:
    tagged = nltk.pos_tag(["a", "very", *tokens, "character"], lang="eng")
    return [tag for _, tag in tagged[2 : 2 + len(tokens)]]


def degree_base_form(token: str, context_pos: str) -> str:
    degree_like = (
        context_pos in {"JJR", "JJS"}
        or re.search(r"(?:er|est|ier|iest)$", token) is not None
        or token in wn._exception_map.get("a", {})
    )
    if not degree_like:
        return token
    for exception in wn._exception_map.get("a", {}).get(token, []):
        name = exception.replace("_", "-").lower()
        if re.fullmatch(r"[a-z]+(?:-[a-z]+)*", name) and has_adjective_synset(name):
            return name
    synsets = wn.synsets(token, pos=wn.ADJ) or wn.synsets(token, pos=wn.ADJ_SAT)
    for synset in synsets:
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", "-").lower()
            if re.fullmatch(r"[a-z]+(?:-[a-z]+)*", name):
                return name
    return token


def remove_degree_forms(tokens: list[str]) -> tuple[list[str], bool]:
    context_pos = pos_in_adjective_context(tokens)
    base_tokens = [degree_base_form(token, tag) for token, tag in zip(tokens, context_pos, strict=True)]
    return base_tokens, base_tokens != tokens


def spelling_base_form(token: str) -> str:
    candidates = []
    if token.endswith("ised"):
        candidates.extend([f"{token[:-4]}ized", f"{token[:-4]}iced"])
    if token.endswith("ising"):
        candidates.extend([f"{token[:-5]}izing"])
    for candidate in candidates:
        if has_adjective_synset(candidate):
            return candidate
    return token


def normalize_spelling_forms(tokens: list[str]) -> tuple[list[str], bool]:
    base_tokens = [spelling_base_form(token) for token in tokens]
    return base_tokens, base_tokens != tokens


def is_temporary_participial_state(value: str) -> bool:
    tokens = tokens_for(value)
    if len(tokens) != 1:
        return False
    synsets = wn.synsets(value, pos=wn.ADJ) + wn.synsets(value, pos=wn.ADJ_SAT)
    if not synsets:
        return False
    definitions = " ".join(synset.definition().lower() for synset in synsets[:4])
    return any(marker in definitions for marker in TEMPORARY_STATE_MARKERS)


def is_domain_only_adjective(value: str) -> bool:
    synsets = wn.synsets(value, pos=wn.ADJ) + wn.synsets(value, pos=wn.ADJ_SAT)
    if not synsets:
        return False
    definitions = " ".join(synset.definition().lower() for synset in synsets[:4])
    return any(marker in definitions for marker in DOMAIN_ONLY_DEFINITION_MARKERS)


def token_maps_adjectivally(token: str, context_pos: str) -> bool:
    if has_adjective_synset(token):
        return True
    if context_pos in {"JJ", "JJR", "JJS"} and not has_nonadjective_synset(token):
        return True
    return False


def adjectival_canonical(value: str, max_words: int) -> tuple[str | None, str]:
    normalized = normalize_tag(value)
    if not normalized:
        return None, "empty_after_normalization"
    if any(ord(char) > 127 for char in normalized):
        return None, "non_ascii"
    if normalized in GENERIC_NON_PERSONALITY:
        return None, "generic_non_personality"
    tokens = tokens_for(normalized)
    if not tokens:
        return None, "empty_after_tokenization"
    if len(tokens) > max_words:
        return None, "too_many_words"
    if any(token in FUNCTION_WORDS for token in tokens):
        return None, "function_word"
    if normalized in LEGAL_OR_DOMAIN_STATUS_WORDS:
        return None, "legal_or_domain_status"
    tokens, spelling_changed = normalize_spelling_forms(tokens)
    tokens, degree_changed = remove_degree_forms(tokens)
    normalized = " ".join(tokens)
    if is_domain_only_adjective(normalized):
        return None, "domain_only_adjective"
    if is_temporary_participial_state(normalized):
        return None, "temporary_participial_state"

    full = full_phrase_adjective(normalized)
    if full:
        if spelling_changed:
            return full, "spelling_canonical_wordnet_adjective"
        return full, "degree_canonical_wordnet_adjective" if degree_changed else "full_phrase_wordnet_adjective"

    context_pos = pos_in_adjective_context(tokens)
    if all(token_maps_adjectivally(token, tag) for token, tag in zip(tokens, context_pos, strict=True)):
        if spelling_changed:
            return " ".join(tokens), "spelling_canonical_tokenwise_adjectival"
        return " ".join(tokens), "degree_canonical_tokenwise_adjectival" if degree_changed else "tokenwise_adjectival"

    return None, "not_adjectival"


def iter_anilist_description(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for character in payload.get("characters", []):
        for tag in (character.get("llm_tags") or {}).get("personality") or []:
            rows.append(
                {
                    "source": "anilist_description_qwen",
                    "anilist_character_id": character.get("anilist_character_id"),
                    "name": character.get("name"),
                    "first_anime": character.get("first_anime"),
                    "tag": tag.get("tag"),
                    "confidence": tag.get("confidence"),
                    "evidence": tag.get("evidence"),
                }
            )
    return rows


def iter_jsonl_personalities(path: Path, source_name: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            character = json.loads(line)
            for tag in (character.get("tags") or {}).get("personality") or []:
                rows.append(
                    {
                        "source": source_name,
                        "anilist_character_id": character.get("anilist_character_id"),
                        "name": character.get("name"),
                        "first_anime": character.get("first_anime"),
                        "tag": tag.get("tag"),
                        "confidence": tag.get("confidence"),
                        "evidence": tag.get("evidence"),
                        "source_key": tag.get("source_key"),
                        "source_url": tag.get("source_url"),
                    }
                )
    return rows


def iter_vndb_personalities(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for character in payload.get("characters", []):
        seen: set[str] = set()
        for tag in (character.get("llm_tags") or {}).get("personality") or []:
            sources = {str(source).lower() for source in (tag.get("sources") or [])}
            if "vndb" not in sources:
                continue
            groups = [str(group) for group in (tag.get("vndb_groups") or [])]
            if any(group.lower() == "role" for group in groups):
                continue
            value = str(tag.get("tag") or "").strip()
            key = value.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": "vndb_personality_traits",
                    "anilist_character_id": character.get("anilist_character_id"),
                    "name": character.get("name"),
                    "first_anime": character.get("first_anime"),
                    "tag": value,
                    "confidence": tag.get("confidence") or "vndb",
                    "evidence": "VNDB character trait"
                    + (f" ({', '.join(groups)})" if groups else ""),
                    "vndb_groups": groups,
                    "vndb_char_count": tag.get("vndb_char_count"),
                }
            )
    return rows


def load_assignments() -> list[dict[str, Any]]:
    assignments = []
    assignments.extend(iter_anilist_description(DEFAULT_SOURCES["anilist_description_qwen"]))
    assignments.extend(iter_vndb_personalities(DEFAULT_SOURCES["vndb_personality_traits"]))
    assignments.extend(iter_jsonl_personalities(DEFAULT_SOURCES["bangumi_local_ollama_qwen"], "bangumi_local_ollama_qwen"))
    assignments.extend(iter_jsonl_personalities(DEFAULT_SOURCES["bangumi_a100_batch_qwen"], "bangumi_a100_batch_qwen"))
    return assignments


def main() -> None:
    args = parse_args()
    raw_to_canonical = load_raw_to_canonical(args.global_canonicalization)
    raw_assignments = load_assignments()
    kept_assignments: list[dict[str, Any]] = []
    reject_reasons = Counter()
    accept_reasons = Counter()
    source_raw_counts = Counter()
    source_kept_counts = Counter()

    for assignment in raw_assignments:
        source_raw_counts[assignment["source"]] += 1
        raw_tag = normalize_tag(str(assignment.get("tag") or ""))
        mapped_tag = raw_to_canonical.get(raw_tag, raw_tag)
        canonical, reason = adjectival_canonical(mapped_tag, args.max_words)
        if not canonical:
            reject_reasons[reason] += 1
            continue
        accept_reasons[reason] += 1
        source_kept_counts[assignment["source"]] += 1
        kept_assignments.append(
            {
                **assignment,
                "original_tag": assignment.get("tag"),
                "tag": canonical,
                "adjectival_mapping": reason,
            }
        )

    by_tag: dict[str, dict[str, Any]] = {}
    for assignment in kept_assignments:
        tag = assignment["tag"]
        row = by_tag.setdefault(
            tag,
            {
                "tag": tag,
                "assignment_count": 0,
                "character_count": 0,
                "sources": Counter(),
                "original_tags": Counter(),
                "mapping_reasons": Counter(),
                "examples": [],
                "_characters": set(),
            },
        )
        row["assignment_count"] += 1
        row["sources"][assignment["source"]] += 1
        row["original_tags"][str(assignment.get("original_tag") or "")] += 1
        row["mapping_reasons"][assignment["adjectival_mapping"]] += 1
        if assignment.get("anilist_character_id") is not None:
            row["_characters"].add(int(assignment["anilist_character_id"]))
        if len(row["examples"]) < 5:
            row["examples"].append(
                {
                    "source": assignment["source"],
                    "character_id": assignment.get("anilist_character_id"),
                    "name": assignment.get("name"),
                    "first_anime": assignment.get("first_anime"),
                    "original_tag": assignment.get("original_tag"),
                    "evidence": assignment.get("evidence"),
                }
            )

    descriptors = []
    for row in by_tag.values():
        row = dict(row)
        characters = row.pop("_characters")
        row["character_count"] = len(characters)
        row["sources"] = dict(row["sources"])
        row["original_tags"] = row["original_tags"].most_common(12)
        row["mapping_reasons"] = dict(row["mapping_reasons"])
        descriptors.append(row)
    descriptors.sort(key=lambda row: (-row["character_count"], -row["assignment_count"], row["tag"]))

    args.assignments_output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.assignments_output.with_name(f"{args.assignments_output.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for assignment in kept_assignments:
            handle.write(json.dumps(assignment, ensure_ascii=False) + "\n")
    tmp.replace(args.assignments_output)

    payload = {
        "generated_at": utc_now(),
        "sources": {name: str(path) for name, path in DEFAULT_SOURCES.items()},
        "parameters": {
            "category": "personality",
            "max_words": args.max_words,
            "adjectival_gate": "WordNet full-phrase adjective OR tokenwise WordNet adjective/unknown-JJ with no non-adjective WordNet sense",
        },
        "counts": {
            "raw_assignments": len(raw_assignments),
            "kept_assignments": len(kept_assignments),
            "unique_adjectival_descriptors": len(descriptors),
            "source_raw_assignments": dict(source_raw_counts),
            "source_kept_assignments": dict(source_kept_counts),
            "accept_reasons": dict(accept_reasons),
            "reject_reasons": dict(reject_reasons),
        },
        "descriptors": descriptors,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.assignments_output}")


if __name__ == "__main__":
    main()
