#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact seiyuu role cache from AniList credit dumps.")
    parser.add_argument("--credit-splits-input", type=Path, required=True)
    parser.add_argument("--characters-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/role_edges.json"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def character_name(row: dict) -> str:
    name = row.get("name")
    if isinstance(name, dict):
        return str(name.get("full") or name.get("userPreferred") or name.get("romaji") or "")
    return str(name or "")


def title_from_row(row: object) -> str:
    if not isinstance(row, dict):
        return ""
    title = row.get("title")
    if isinstance(title, dict):
        return str(title.get("english") or title.get("romaji") or title.get("native") or "")
    return str(title or "")


def anilist_staff_url(seiyuu_id: int) -> str:
    return f"https://anilist.co/staff/{seiyuu_id}"


def anilist_anime_url(anime_id: int | None) -> str:
    return f"https://anilist.co/anime/{anime_id}" if anime_id else ""


def mal_anime_url(mal_id: int | None) -> str:
    return f"https://myanimelist.net/anime/{mal_id}" if mal_id else ""


def build_character_map(payload: dict) -> dict[int, dict]:
    output: dict[int, dict] = {}
    for row in payload.get("characters", []):
        character_id = int(row["id"])
        output[character_id] = {
            "character_id": character_id,
            "name": character_name(row),
            "native_name": row.get("native_name") or "",
            "gender": row.get("gender") or "",
            "favourites": int(row.get("favourites") or 0),
            "image": row.get("image") or "",
            "site_url": row.get("site_url") or f"https://anilist.co/character/{character_id}",
            "first_anime": title_from_row(row.get("window_first_anime") or row.get("first_anime")),
        }
    return output


def build_cache(credit_payload: dict, characters_payload: dict) -> dict:
    character_map = build_character_map(characters_payload)
    seiyuu_by_id = {
        int(row["seiyuu_id"]): {
            "seiyuu_id": int(row["seiyuu_id"]),
            "name": row.get("name") or "",
            "native_name": row.get("native_name") or "",
            "language": row.get("language") or "",
            "image": row.get("image") or "",
            "site_url": row.get("site_url") or anilist_staff_url(int(row["seiyuu_id"])),
        }
        for row in credit_payload.get("seiyuu", [])
    }

    grouped: dict[tuple[int, int], dict] = {}
    total_credit_edges = 0
    for split, edges in credit_payload.get("credit_edges_by_split", {}).items():
        for edge in edges:
            total_credit_edges += 1
            seiyuu_id = int(edge["seiyuu_id"])
            character_id = int(edge["character_id"])
            seiyuu = seiyuu_by_id.get(
                seiyuu_id,
                {
                    "seiyuu_id": seiyuu_id,
                    "name": "",
                    "native_name": "",
                    "language": "JAPANESE",
                    "image": "",
                    "site_url": anilist_staff_url(seiyuu_id),
                },
            )
            character = character_map.get(
                character_id,
                {
                    "character_id": character_id,
                    "name": edge.get("character_name") or "",
                    "native_name": edge.get("character_native_name") or "",
                    "gender": edge.get("character_gender") or "",
                    "favourites": 0,
                    "image": "",
                    "site_url": f"https://anilist.co/character/{character_id}",
                    "first_anime": "",
                },
            )
            key = (seiyuu_id, character_id)
            row = grouped.setdefault(
                key,
                {
                    "seiyuu": seiyuu,
                    "character": character,
                    "character_role": edge.get("character_role") or "",
                    "anime": [],
                    "first_year": None,
                    "latest_year": None,
                    "credit_count": 0,
                    "splits": [],
                },
            )
            year = edge.get("anime_start_year") or edge.get("anime_year") or edge.get("anime_season_year")
            if year:
                row["first_year"] = int(year) if row["first_year"] is None else min(row["first_year"], int(year))
                row["latest_year"] = int(year) if row["latest_year"] is None else max(row["latest_year"], int(year))
            anime_id = edge.get("anime_id")
            mal_id = edge.get("mal_anime_id")
            row["anime"].append(
                {
                    "anime_id": anime_id,
                    "title": edge.get("anime_title") or "",
                    "year": int(year) if year else None,
                    "site_url": anilist_anime_url(anime_id),
                    "mal_url": mal_anime_url(mal_id),
                }
            )
            row["credit_count"] += 1
            row["splits"].append(split)

    seiyuu_counts: dict[int, dict] = defaultdict(lambda: {"role_count": 0, "character_count": 0, "first_year": None})
    for row in grouped.values():
        seiyuu_id = row["seiyuu"]["seiyuu_id"]
        seiyuu_counts[seiyuu_id]["role_count"] += int(row["credit_count"])
        seiyuu_counts[seiyuu_id]["character_count"] += 1
        if row["first_year"]:
            current = seiyuu_counts[seiyuu_id]["first_year"]
            seiyuu_counts[seiyuu_id]["first_year"] = row["first_year"] if current is None else min(current, row["first_year"])

    for row in grouped.values():
        counts = seiyuu_counts[row["seiyuu"]["seiyuu_id"]]
        row["seiyuu"] = {**row["seiyuu"], **counts}
        row["splits"] = sorted(set(row["splits"]))
        row["anime"] = sorted(row["anime"], key=lambda anime: (anime["year"] or 9999, anime["title"]))[:8]

    return {
        "generated_at": utc_now(),
        "source": "prepare_compact_cache.py",
        "counts": {
            "credit_edges": total_credit_edges,
            "seiyuu_character_pairs": len(grouped),
            "seiyuu": len(seiyuu_counts),
            "characters": len({key[1] for key in grouped}),
        },
        "roles": sorted(grouped.values(), key=lambda row: (row["seiyuu"]["name"], row["character"]["name"])),
    }


def main() -> None:
    args = parse_args()
    payload = build_cache(read_json(args.credit_splits_input), read_json(args.characters_input))
    write_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
