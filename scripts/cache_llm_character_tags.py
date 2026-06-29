#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fit_topics import import_nltk


TAG_CATEGORIES = ("role", "personality", "traits")
NOUN_POS_TAGS = {"NN", "NNS", "NNP", "NNPS"}
ADJECTIVE_POS_TAGS = {"JJ", "JJR", "JJS"}
ADJECTIVE_LIKE_POS_TAGS = {"VBG", "VBN"}
UNKNOWN_DESCRIPTOR_POS_TAGS = {"FW", "EX", "RB"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured character tags from descriptions with Ollama.")
    parser.add_argument("--roles-input", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--descriptions-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/external/llm/character_description_tags.json"))
    parser.add_argument("--raw-cache-dir", type=Path, default=Path("data/external/llm/raw_character_tags"))
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--min-favourites", type=int, default=100)
    parser.add_argument("--min-description-words", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument(
        "--think",
        choices=("false", "true", "low", "medium", "high", "max"),
        default="false",
        help="Ollama thinking mode. Defaults to false for faster schema extraction.",
    )
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--seiyuu-name", action="append", default=[])
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def normalize_name(value: str) -> str:
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9 ]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def name_keys(value: str) -> set[str]:
    norm = normalize_name(value)
    if not norm:
        return set()
    parts = norm.split()
    return {norm, " ".join(reversed(parts)), " ".join(sorted(parts))}


def normalize_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", value)
    value = re.sub(r"https?://\S+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def description_lookup(payload: Any) -> dict[str, str]:
    if isinstance(payload, dict) and isinstance(payload.get("characters"), list):
        return {
            str(row.get("id") or row.get("character_id")): str(
                row.get("description") or row.get("description_plain") or row.get("description_text") or ""
            )
            for row in payload["characters"]
        }
    if isinstance(payload, dict):
        return {
            str(key): str(
                value.get("description") or value.get("description_plain") or value.get("description_text") or ""
            )
            if isinstance(value, dict)
            else str(value or "")
            for key, value in payload.items()
        }
    raise TypeError("Descriptions input must be a JSON object or an object with a characters list.")


def unique_characters(roles: list[dict], descriptions: dict[str, str], args: argparse.Namespace) -> list[dict]:
    by_id = {}
    requested_names = set().union(*(name_keys(value) for value in args.name)) if args.name else None
    requested_seiyuu_names = set().union(*(name_keys(value) for value in args.seiyuu_name)) if args.seiyuu_name else None
    for role in roles:
        seiyuu = role.get("seiyuu", {})
        if requested_seiyuu_names and not name_keys(seiyuu.get("name") or "").intersection(requested_seiyuu_names):
            continue
        character = role.get("character", {})
        if int(character.get("favourites") or 0) < args.min_favourites:
            continue
        if requested_names and not name_keys(character.get("name") or "").intersection(requested_names):
            continue
        character_id = int(character["character_id"])
        description = normalize_text(descriptions.get(str(character_id), ""))
        if len(description.split()) < args.min_description_words:
            continue
        by_id.setdefault(
            character_id,
            {
                "anilist_character_id": character_id,
                "name": character.get("name") or "",
                "native_name": character.get("native_name") or "",
                "gender": character.get("gender") or "",
                "first_anime": character.get("first_anime") or "",
                "favourites": int(character.get("favourites") or 0),
                "site_url": character.get("site_url") or "",
                "seiyuu": [],
                "description": description,
            },
        )
        seiyuu_key = {
            "seiyuu_id": seiyuu.get("seiyuu_id"),
            "name": seiyuu.get("name") or "",
            "native_name": seiyuu.get("native_name") or "",
        }
        if seiyuu_key not in by_id[character_id]["seiyuu"]:
            by_id[character_id]["seiyuu"].append(seiyuu_key)
    return sorted(by_id.values(), key=lambda row: (-row["favourites"], row["name"]))


def prompt_for_character(row: dict) -> str:
    return f"""Extract structured tags from this anime character description.

Use only the provided description. Do not use outside knowledge, anime fandom knowledge, image cues, or the character name.
Every tag must be directly supported by an exact evidence span copied from the description.

Categories:
- role: social, story, family, school, job, team, relationship, or narrative roles.
- personality: stable temperament, attitude, behavior style, or interpersonal style.
- traits: other stable non-appearance traits, abilities, interests, habits, skills, or conditions.

Rules:
- Each category is an array and may contain zero, one, or many tags.
- The tag field must be a canonical phrase of 1 to 4 words.
- Tag words must be only adjectives and nouns.
- Do not use pronouns, verbs, conjunctions, prepositions, determiners, particles, or adverbs in tags.
- If a concept combines multiple tags with "and" or "/", split it into multiple entries.
- Convert supported verb clauses into adjective/noun tags when possible.
- Examples: "can't say no to people who need help" -> "helpful"; "aspires to become an actress" -> "aspiring actress"; "acts as the mother of the family" -> "mother figure".
- Do not include hair, eye, outfit, body-shape, or other appearance tags.
- Do not include numeric measurements such as height, weight, bust, waist, or hip sizes.
- Do not include plot-only events unless they imply a stable role or trait.
- Do not include duplicate or near-duplicate tags.
- If a category has no directly supported tags, return an empty array.
- If uncertain, omit.

Character metadata:
- anime: {row['first_anime']}
- gender: {row['gender']}

Description:
{row['description']}

Return only JSON with this shape:
{{
  "role": [
    {{"tag": "short tag", "evidence": "exact text span", "confidence": "high|medium|low"}}
  ],
  "personality": [
    {{"tag": "short tag", "evidence": "exact text span", "confidence": "high|medium|low"}}
  ],
  "traits": [
    {{"tag": "short tag", "evidence": "exact text span", "confidence": "high|medium|low"}}
  ]
}}
"""


def parse_json_object(value: str) -> dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            return json.loads(value[start : end + 1])
        raise


def raw_cache_path(raw_cache_dir: Path, row: dict) -> Path:
    return raw_cache_dir / f"{row['anilist_character_id']}.json"


def call_ollama(args: argparse.Namespace, prompt: str) -> dict:
    think: bool | str = args.think
    if args.think == "false":
        think = False
    elif args.think == "true":
        think = True
    payload = {
        "model": args.ollama_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict information-extraction assistant. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "think": think,
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
    }
    request = urllib.request.Request(
        f"{args.ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Ollama at {args.ollama_url}.") from error


def cached_llm_response(row: dict, args: argparse.Namespace) -> dict:
    path = raw_cache_path(args.raw_cache_dir, row)
    if path.exists() and not args.refresh_raw:
        return read_json(path)
    if args.offline:
        raise FileNotFoundError(f"missing raw LLM cache for {row['name']} at {path}")
    prompt = prompt_for_character(row)
    response = call_ollama(args, prompt)
    cached = {
        "generated_at": utc_now(),
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "think": args.think,
        },
        "local_character": {key: value for key, value in row.items() if key != "description"},
        "prompt": prompt,
        "response": response,
    }
    write_json(path, cached)
    return cached


def evidence_supported(evidence: str, description: str) -> bool:
    if not evidence:
        return False
    evidence_text = normalize_evidence(evidence).lower()
    description_text = normalize_text(description).lower()
    if evidence_text in description_text:
        return True

    # Qwen often returns copied spans with leading/trailing/interior ellipses.
    # Keep validation strict by requiring every nontrivial fragment to appear in
    # order in the source description.
    fragments = [
        normalize_text(fragment).lower().strip(" .")
        for fragment in re.split(r"(?:\.\.\.|…)+", evidence_text)
    ]
    fragments = [fragment for fragment in fragments if len(fragment) >= 4]
    if not fragments:
        return False
    cursor = 0
    for fragment in fragments:
        index = description_text.find(fragment, cursor)
        if index < 0:
            return False
        cursor = index + len(fragment)
    return True


def normalize_evidence(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return value.strip("\"'“”‘’")


def tag_tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]*", value)]


def adjective_like_token(token: str) -> bool:
    try:
        from nltk.corpus import wordnet as wn
        return bool(wn.synsets(token.replace("-", "_"), pos=wn.ADJ))
    except LookupError:
        return False


def unknown_descriptor_token(token: str) -> bool:
    if not re.fullmatch(r"[a-z][a-z-]*", token) or token.endswith("ly"):
        return False
    try:
        from nltk.corpus import wordnet as wn
        return not wn.synsets(token.replace("-", "_"))
    except LookupError:
        return False


def finite_verb_like_descriptor_head(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    head = tokens[0]
    if not (head.endswith("s") or head.endswith("ed")):
        return False
    try:
        from nltk.corpus import wordnet as wn

        return bool(wn.morphy(head, wn.VERB))
    except LookupError:
        return False


def has_possessive_descriptor_token(tokens: list[str]) -> bool:
    return any(token.endswith("'s") or token.endswith("’s") for token in tokens)


def canonical_tag(value: str, nltk: Any) -> str | None:
    if re.search(r"\d", value):
        return None
    tokens = tag_tokens(value)
    if not tokens or len(tokens) > 4:
        return None
    if has_possessive_descriptor_token(tokens):
        return None
    if finite_verb_like_descriptor_head(tokens):
        return None
    tagged = nltk.pos_tag(tokens)
    for token, tag in tagged:
        if tag in NOUN_POS_TAGS or tag in ADJECTIVE_POS_TAGS:
            continue
        if tag in ADJECTIVE_LIKE_POS_TAGS and adjective_like_token(token):
            continue
        if tag in UNKNOWN_DESCRIPTOR_POS_TAGS and unknown_descriptor_token(token):
            continue
        return None
    return " ".join(tokens)


def validate_tags(raw: dict, description: str, nltk: Any) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {category: [] for category in TAG_CATEGORIES}
    seen: set[tuple[str, str]] = set()
    for category in TAG_CATEGORIES:
        values = raw.get(category, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            tag = canonical_tag(str(value.get("tag", "")), nltk)
            evidence = normalize_evidence(str(value.get("evidence", "")))
            confidence = str(value.get("confidence", "medium")).strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "medium"
            if not tag or not evidence_supported(evidence, description):
                continue
            key = (category, tag)
            if key in seen:
                continue
            seen.add(key)
            output[category].append({"tag": tag, "evidence": evidence, "confidence": confidence})
    return output


def extract_content(cached: dict) -> dict:
    message = cached.get("response", {}).get("message", {})
    content = message.get("content", "")
    if not content and message.get("thinking"):
        content = str(message["thinking"])
    if not content:
        raise RuntimeError("Ollama returned no content.")
    return parse_json_object(content)


def build_payload(args: argparse.Namespace, characters: list[dict], rows: dict[int, dict], requested: int) -> dict:
    values_by_category: dict[str, set[str]] = {category: set() for category in TAG_CATEGORIES}
    tagged_characters = 0
    tag_counts = {category: 0 for category in TAG_CATEGORIES}
    for row in rows.values():
        has_any = False
        for category in TAG_CATEGORIES:
            tags = row.get("llm_tags", {}).get(category, [])
            if tags:
                has_any = True
            tag_counts[category] += len(tags)
            for tag in tags:
                values_by_category[category].add(tag["tag"])
        if has_any:
            tagged_characters += 1

    return {
        "generated_at": utc_now(),
        "source": "cache_llm_character_tags.py",
        "parameters": {
            "roles_input": str(args.roles_input),
            "descriptions_input": str(args.descriptions_input),
            "raw_cache_dir": str(args.raw_cache_dir),
            "ollama_model": args.ollama_model,
            "temperature": args.temperature,
            "seed": args.seed,
            "think": args.think,
            "limit": args.limit,
            "min_favourites": args.min_favourites,
            "min_description_words": args.min_description_words,
            "checkpoint_every": args.checkpoint_every,
            "name": args.name,
            "seiyuu_name": args.seiyuu_name,
        },
        "counts": {
            "local_characters": len(characters),
            "queried_this_run": requested,
            "cached_characters": len(rows),
            "tagged_characters": tagged_characters,
            "tag_counts": tag_counts,
        },
        "values_by_category": {key: sorted(value) for key, value in values_by_category.items()},
        "characters": sorted(rows.values(), key=lambda row: (-row.get("favourites", 0), row.get("name", ""))),
    }


def main() -> None:
    args = parse_args()
    nltk = import_nltk()
    roles_payload = read_json(args.roles_input)
    descriptions = description_lookup(read_json(args.descriptions_input))
    characters = unique_characters(roles_payload.get("roles", []), descriptions, args)
    existing = read_json(args.output) if args.output.exists() and not args.force else {}
    existing_rows = {int(row["anilist_character_id"]): row for row in existing.get("characters", [])}

    pending = [row for row in characters if row["anilist_character_id"] not in existing_rows]
    if args.limit is not None and args.limit > 0:
        pending = pending[: args.limit]
    if args.dry_run:
        if not pending:
            raise RuntimeError("No pending characters with descriptions.")
        print(prompt_for_character(pending[0]))
        return

    rows = dict(existing_rows)
    requested = 0
    for row in pending:
        requested += 1
        try:
            cached = cached_llm_response(row, args)
            raw_tags = extract_content(cached)
            tags = validate_tags(raw_tags, row["description"], nltk)
            rows[row["anilist_character_id"]] = {
                **{key: value for key, value in row.items() if key != "description"},
                "llm_tags": tags,
                "llm_raw_cache": str(raw_cache_path(args.raw_cache_dir, row)),
            }
        except Exception as error:
            rows[row["anilist_character_id"]] = {
                **{key: value for key, value in row.items() if key != "description"},
                "llm_tags": {category: [] for category in TAG_CATEGORIES},
                "llm_raw_cache": str(raw_cache_path(args.raw_cache_dir, row)),
                "error": str(error)[:500],
            }
        if args.checkpoint_every and requested % args.checkpoint_every == 0:
            write_json(args.output, build_payload(args, characters, rows, requested))
            print(f"checkpoint {requested}/{len(pending)} -> {args.output}", flush=True)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    payload = build_payload(args, characters, rows, requested)
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
