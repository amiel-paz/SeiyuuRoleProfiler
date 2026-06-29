#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VNDB_CHARACTER_ENDPOINT = "https://api.vndb.org/kana/character"
CHARACTER_FIELDS = (
    "name,original,aliases,description,"
    "vns{title,role},"
    "traits{name,group_name,spoiler,lie,sexual,char_count}"
)
TITLE_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "season",
    "movie",
    "part",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache VNDB character traits for local AniList role characters.")
    parser.add_argument("--roles-input", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--output", type=Path, default=Path("data/external/vndb/vndb_character_traits.json"))
    parser.add_argument("--raw-cache-dir", type=Path, default=Path("data/external/vndb/raw_character_search"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--min-favourites", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=1.6)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-spoiler", type=int, default=0)
    parser.add_argument("--include-sexual", action="store_true")
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Restrict to local character names matching this value. Can be repeated.",
    )
    parser.add_argument("--force", action="store_true")
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


def title_tokens(value: str) -> set[str]:
    return {token for token in normalize_name(value).split() if len(token) >= 3 and token not in TITLE_STOPWORDS}


def unique_characters(roles: list[dict], min_favourites: int) -> list[dict]:
    by_id = {}
    for role in roles:
        character = role.get("character", {})
        if int(character.get("favourites") or 0) < min_favourites:
            continue
        character_id = int(character["character_id"])
        by_id.setdefault(
            character_id,
            {
                "anilist_character_id": character_id,
                "name": character.get("name") or "",
                "native_name": character.get("native_name") or "",
                "first_anime": character.get("first_anime") or "",
                "favourites": int(character.get("favourites") or 0),
                "site_url": character.get("site_url") or "",
            },
        )
    return sorted(by_id.values(), key=lambda row: (-row["favourites"], row["name"]))


def post_vndb(payload: dict) -> dict:
    request = urllib.request.Request(
        VNDB_CHARACTER_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SeiyuuRoleProfiler/0.1 non-commercial research cache",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def search_payload(name: str) -> dict:
    return {
        "filters": ["search", "=", name],
        "fields": CHARACTER_FIELDS,
        "sort": "searchrank",
        "results": 10,
    }


def raw_cache_path(raw_cache_dir: Path, local: dict) -> Path:
    return raw_cache_dir / f"{local['anilist_character_id']}.json"


def cached_vndb_search(local: dict, args: argparse.Namespace) -> dict:
    path = raw_cache_path(args.raw_cache_dir, local)
    if path.exists() and not args.refresh_raw:
        return read_json(path)
    if args.offline:
        raise FileNotFoundError(f"missing raw VNDB cache for {local['name']} at {path}")

    payload = search_payload(local["name"])
    response = post_vndb(payload)
    cached = {
        "generated_at": utc_now(),
        "endpoint": VNDB_CHARACTER_ENDPOINT,
        "request": payload,
        "local_character": local,
        "response": response,
    }
    write_json(path, cached)
    return cached


def filter_traits(traits: list[dict], max_spoiler: int, include_sexual: bool) -> list[dict]:
    output = []
    for trait in traits:
        if trait.get("lie"):
            continue
        if int(trait.get("spoiler") or 0) > max_spoiler:
            continue
        if trait.get("sexual") and not include_sexual:
            continue
        output.append(
            {
                "group": trait.get("group_name") or "",
                "name": trait.get("name") or "",
                "spoiler": int(trait.get("spoiler") or 0),
                "sexual": bool(trait.get("sexual")),
                "char_count": int(trait.get("char_count") or 0),
            }
        )
    return output


def choose_best_match(local: dict, candidates: list[dict], max_spoiler: int, include_sexual: bool) -> dict | None:
    local_keys = name_keys(local["name"])
    local_token_count = len(normalize_name(local["name"]).split())
    local_title_tokens = title_tokens(local.get("first_anime") or "")
    scored = []
    for candidate in candidates:
        candidate_keys = name_keys(candidate.get("name") or "")
        if candidate.get("original"):
            candidate_keys.update(name_keys(candidate["original"]))
        for alias in candidate.get("aliases") or []:
            candidate_keys.update(name_keys(alias))
        exact = bool(local_keys.intersection(candidate_keys))
        if not exact:
            continue
        candidate_title_tokens = set()
        for vn in candidate.get("vns") or []:
            candidate_title_tokens.update(title_tokens(vn.get("title") or ""))
        title_overlap = len(local_title_tokens.intersection(candidate_title_tokens))
        if local_token_count < 2 and local_title_tokens and title_overlap == 0:
            continue
        traits = filter_traits(candidate.get("traits") or [], max_spoiler, include_sexual)
        scored.append(
            (
                title_overlap,
                len(traits),
                candidate.get("id") or "",
                {
                    "vndb_character_id": candidate.get("id"),
                    "name": candidate.get("name") or "",
                    "original": candidate.get("original"),
                    "match": "exact_alias_or_reordered_name",
                    "title_token_overlap": title_overlap,
                    "vns": candidate.get("vns") or [],
                    "traits": traits,
                },
            )
        )
    if not scored:
        return None
    return sorted(scored, key=lambda item: (-item[1], -item[0], item[2]))[0][3]


def build_payload(args: argparse.Namespace, characters: list[dict], rows: dict[int, dict], requested: int) -> dict:
    values_by_group: dict[str, set[str]] = {}
    matched = 0
    with_traits = 0
    for row in rows.values():
        match = row.get("vndb_best_match")
        if not match:
            continue
        matched += 1
        traits = match.get("traits") or []
        if traits:
            with_traits += 1
        for trait in traits:
            values_by_group.setdefault(trait["group"], set()).add(trait["name"])

    return {
        "generated_at": utc_now(),
        "source": "cache_vndb_character_traits.py",
        "parameters": {
            "roles_input": str(args.roles_input),
            "raw_cache_dir": str(args.raw_cache_dir),
            "limit": args.limit,
            "min_favourites": args.min_favourites,
            "sleep_seconds": args.sleep_seconds,
            "checkpoint_every": args.checkpoint_every,
            "max_spoiler": args.max_spoiler,
            "include_sexual": args.include_sexual,
            "refresh_raw": args.refresh_raw,
            "offline": args.offline,
            "name": args.name,
        },
        "counts": {
            "local_characters": len(characters),
            "queried_this_run": requested,
            "cached_characters": len(rows),
            "vndb_matches": matched,
            "vndb_matches_with_traits": with_traits,
        },
        "values_by_group": {key: sorted(value) for key, value in sorted(values_by_group.items())},
        "characters": sorted(rows.values(), key=lambda row: (-row.get("favourites", 0), row.get("name", ""))),
    }


def main() -> None:
    args = parse_args()
    roles_payload = read_json(args.roles_input)
    existing = read_json(args.output) if args.output.exists() and not args.force else {}
    existing_rows = {int(row["anilist_character_id"]): row for row in existing.get("characters", [])}

    characters = unique_characters(roles_payload.get("roles", []), args.min_favourites)
    if args.name:
        requested_names = set().union(*(name_keys(value) for value in args.name))
        characters = [row for row in characters if name_keys(row["name"]).intersection(requested_names)]
    pending = [row for row in characters if row["anilist_character_id"] not in existing_rows]
    if args.limit is not None and args.limit > 0:
        pending = pending[: args.limit]

    rows = dict(existing_rows)
    requested = 0
    for local in pending:
        requested += 1
        try:
            raw = cached_vndb_search(local, args)
            candidates = raw.get("response", {}).get("results", [])
            best = choose_best_match(local, candidates, args.max_spoiler, args.include_sexual)
            rows[local["anilist_character_id"]] = {
                **local,
                "vndb_best_match": best,
                "vndb_candidate_count": len(candidates),
                "vndb_raw_cache": str(raw_cache_path(args.raw_cache_dir, local)),
            }
        except Exception as error:
            rows[local["anilist_character_id"]] = {
                **local,
                "vndb_best_match": None,
                "vndb_candidate_count": 0,
                "vndb_raw_cache": str(raw_cache_path(args.raw_cache_dir, local)),
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
    main()
