#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INHERITANCE_WEIGHT = 0.25

KINSHIP_OR_GENERIC_TARGET_WORDS = {
    "father",
    "fathers",
    "mother",
    "mothers",
    "parent",
    "parents",
    "brother",
    "brothers",
    "sister",
    "sisters",
    "grandfather",
    "grandmother",
    "temperament",
    "personality",
    "attitude",
}

REFERENCE_STRIP_WORDS = {
    "a",
    "an",
    "the",
    "original",
    "alternate",
    "alternative",
    "counterpart",
    "version",
    "universe",
    "directly",
    "character",
}

CONTEXT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "arc",
    "episode",
    "film",
    "is",
    "movie",
    "no",
    "of",
    "ova",
    "season",
    "special",
    "stage",
    "the",
    "to",
    "tv",
    "wa",
}

COMPARATIVE_PERSONALITY_PATTERNS = [
    re.compile(
        r"\b(?:resembles|is\s+similar\s+to|appears\s+to\s+be\s+(?:very\s+)?similar\s+to|"
        r"seems\s+(?:very\s+)?similar\s+to|similar\s+to)\s+"
        r"(?P<target>[^.;!\n]{1,120}?)\s+"
        r"(?:in\s+(?:both\s+[^.;!\n]{0,80}?\s+and\s+)?personality|personality)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpersonality\s+(?:resembles|is\s+similar\s+to|is\s+(?:very\s+)?like)\s+"
        r"(?P<target>[^.;!\n]{1,120})",
        re.IGNORECASE,
    ),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(value: str) -> str:
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9 ]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def name_keys(value: str) -> set[str]:
    normalized = normalize_name(value)
    if not normalized:
        return set()
    parts = normalized.split()
    keys = {normalized}
    if len(parts) >= 2:
        keys.add(" ".join(reversed(parts)))
    return keys


def character_id(character: dict) -> int:
    return int(character.get("anilist_character_id") or character.get("character_id"))


def raw_cache_prompt(character: dict) -> str:
    cache_path = character.get("llm_raw_cache")
    if not cache_path:
        return ""
    path = Path(cache_path)
    if not path.exists():
        return ""
    try:
        return str(read_json(path).get("prompt") or "")
    except (OSError, json.JSONDecodeError):
        return ""


def description_from_raw_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    match = re.search(r"\bDescription:\s*\n(?P<description>.*?)(?:\n\nReturn only JSON|\Z)", prompt, re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group("description")).strip()
    return ""


def source_text(character: dict) -> str:
    direct = str(character.get("description") or "").strip()
    if direct:
        return direct
    return description_from_raw_prompt(raw_cache_prompt(character))


def clean_reference(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"['’]s\b", "", value)
    value = re.sub(r"\b(?:in|from|of)\s+(?:both\s+)?(?:looks|appearance|personality)\b.*$", " ", value, flags=re.I)
    value = re.sub(r"\b(?:parts?|series|anime|manga|universe)s?\b.*$", " ", value, flags=re.I)
    tokens = [token for token in normalize_name(value).split() if token not in REFERENCE_STRIP_WORDS]
    return " ".join(tokens)


def contains_blocked_reference(value: str) -> bool:
    tokens = set(normalize_name(value).split())
    return bool(tokens & KINSHIP_OR_GENERIC_TARGET_WORDS)


def build_character_resolver(characters: list[dict]) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    exact: dict[str, dict] = {}
    token_to_rows: dict[str, list[dict]] = {}
    for character in characters:
        for key in name_keys(str(character.get("name") or "")):
            exact.setdefault(key, character)
        for token in normalize_name(str(character.get("name") or "")).split():
            if len(token) >= 3 and token not in REFERENCE_STRIP_WORDS:
                token_to_rows.setdefault(token, []).append(character)

    return exact, token_to_rows


def context_tokens(value: str) -> set[str]:
    return {
        token
        for token in normalize_name(value).split()
        if len(token) >= 3 and token not in CONTEXT_STOP_WORDS
    }


def disambiguate_single_token_reference(
    token: str,
    source_character: dict,
    candidates_by_id: dict[int, dict],
) -> dict | None:
    if len(candidates_by_id) == 1:
        return next(iter(candidates_by_id.values()))

    source_name_tokens = set(normalize_name(str(source_character.get("name") or "")).split())
    source_context = context_tokens(str(source_character.get("first_anime") or ""))
    source_text_tokens = context_tokens(source_text(source_character))

    scored = []
    for row in candidates_by_id.values():
        candidate_name_tokens = set(normalize_name(str(row.get("name") or "")).split())
        candidate_context = context_tokens(str(row.get("first_anime") or ""))
        score = 0
        score += 4 * len((candidate_name_tokens - {token}) & source_name_tokens)
        score += 3 * len(candidate_context & source_context)
        score += 2 * len((candidate_name_tokens - {token}) & source_text_tokens)
        score += 1 if token in candidate_name_tokens else 0
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def resolve_reference(
    raw_reference: str,
    source_character: dict,
    exact_names: dict[str, dict],
    token_to_rows: dict[str, list[dict]],
) -> dict | None:
    if contains_blocked_reference(raw_reference):
        return None
    cleaned = clean_reference(raw_reference)
    if not cleaned or contains_blocked_reference(cleaned):
        return None
    source_id = character_id(source_character)
    tokens = cleaned.split()
    if len(tokens) == 1 and tokens[0] in token_to_rows:
        candidates_by_id = {
            character_id(row): row for row in token_to_rows[tokens[0]] if character_id(row) != source_id
        }
        resolved = disambiguate_single_token_reference(tokens[0], source_character, candidates_by_id)
        if resolved is not None:
            return resolved
        return None

    if cleaned in exact_names and character_id(exact_names[cleaned]) != source_id:
        return exact_names[cleaned]

    for start in range(len(tokens)):
        for end in range(len(tokens), start, -1):
            candidate = " ".join(tokens[start:end])
            if candidate in exact_names and character_id(exact_names[candidate]) != source_id:
                return exact_names[candidate]
    return None


def comparative_personality_edges(characters: list[dict]) -> list[dict]:
    exact_names, token_to_rows = build_character_resolver(characters)
    edges = []
    for character in characters:
        text = source_text(character)
        if not text:
            continue
        for pattern in COMPARATIVE_PERSONALITY_PATTERNS:
            for match in pattern.finditer(text):
                raw_reference = match.group("target").strip()
                target = resolve_reference(raw_reference, character, exact_names, token_to_rows)
                if target is None:
                    continue
                edge = {
                    "source_character_id": character_id(character),
                    "source_character": character.get("name") or "",
                    "target_character_id": character_id(target),
                    "target_character": target.get("name") or "",
                    "raw_reference": raw_reference,
                    "evidence": re.sub(r"\s+", " ", match.group(0)).strip(),
                }
                if edge not in edges:
                    edges.append(edge)
    return edges


def direct_personality_tags(character: dict) -> list[dict]:
    output = []
    for tag in character.get("llm_tags", {}).get("personality", []):
        value = str(tag.get("tag") or "").strip().lower()
        if not value:
            continue
        if "comparative_personality" in set(tag.get("sources") or []):
            continue
        output.append(tag)
    return output


def inherited_personality_descriptors(
    characters: list[dict],
    weight: float = DEFAULT_INHERITANCE_WEIGHT,
) -> tuple[dict[int, list[dict]], list[dict]]:
    by_id = {character_id(character): character for character in characters}
    edges = comparative_personality_edges(characters)
    inherited: dict[int, list[dict]] = {}
    for edge in edges:
        target = by_id.get(int(edge["target_character_id"]))
        if target is None:
            continue
        source_id = int(edge["source_character_id"])
        for tag in direct_personality_tags(target):
            descriptor = str(tag.get("tag") or "").strip().lower()
            if not descriptor:
                continue
            inherited.setdefault(source_id, []).append(
                {
                    "tag": descriptor,
                    "weight": float(weight),
                    "sources": ["comparative_personality"],
                    "inherited_from_character_id": int(edge["target_character_id"]),
                    "inherited_from_character": edge["target_character"],
                    "comparative_evidence": edge["evidence"],
                }
            )
    return inherited, edges
