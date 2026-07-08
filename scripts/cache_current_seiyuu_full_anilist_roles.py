#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANILIST_GRAPHQL = "https://graphql.anilist.co"
QUERY = """
query ($id: Int!, $page: Int!, $perPage: Int!, $sort: [MediaSort]) {
  Staff(id: $id) {
    id
    name { full native }
    languageV2
    image { medium }
    siteUrl
    characterMedia(page: $page, perPage: $perPage, sort: $sort) {
      pageInfo { currentPage hasNextPage total perPage lastPage }
      edges {
        characterRole
        staffRole
        node {
          id
          idMal
          title { romaji english native }
          startDate { year month day }
          seasonYear
          siteUrl
          type
          format
        }
        characters {
          id
          name { full native }
          gender
          favourites
          image { medium }
          siteUrl
        }
      }
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a seiyuu-first AniList role cache. The seiyuu set comes from the "
            "current profiler manifest, then each staff member's full characterMedia "
            "universe is fetched without a year floor and filtered by character favourites."
        )
    )
    parser.add_argument("--profiles", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--output", type=Path, default=Path("data/role_edges_current_seiyuu_full.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/anilist_staff_roles"))
    parser.add_argument("--min-favourites", type=int, default=100)
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.75)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional staff limit for smoke tests.")
    parser.add_argument("--staff-id", type=int, default=0, help="Optional single AniList staff id for targeted tests.")
    parser.add_argument("--staff-ids-file", type=Path, help="Optional newline-delimited AniList staff ids to fetch.")
    parser.add_argument("--staff-name", default="", help="Optional case-insensitive exact staff name for targeted tests.")
    parser.add_argument(
        "--sorts",
        nargs="+",
        default=["START_DATE", "START_DATE_DESC", "POPULARITY_DESC"],
        help=(
            "Media sort orders to union. Multiple orders help recover both older and newer "
            "roles for high-volume staff when an API connection is capped."
        ),
    )
    parser.add_argument("--refresh", action="store_true", help="Refetch staff JSON even when a raw cache exists.")
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


def title(row: dict) -> str:
    value = row.get("title") or {}
    return str(value.get("english") or value.get("romaji") or value.get("native") or "")


def year(row: dict) -> int | None:
    start = row.get("startDate") or {}
    if start.get("year"):
        return int(start["year"])
    if row.get("seasonYear"):
        return int(row["seasonYear"])
    return None


def staff_rows(profiles_payload: dict) -> list[dict]:
    rows = []
    seen = set()
    for profile in profiles_payload.get("profiles", []):
        seiyuu_id = profile.get("seiyuu_id")
        if seiyuu_id is None:
            continue
        seiyuu_id = int(seiyuu_id)
        if seiyuu_id in seen:
            continue
        seen.add(seiyuu_id)
        rows.append(
            {
                "seiyuu_id": seiyuu_id,
                "name": profile.get("name") or "",
                "native_name": profile.get("native_name") or "",
                "image": profile.get("image") or "",
                "site_url": profile.get("site_url") or f"https://anilist.co/staff/{seiyuu_id}",
            }
        )
    return rows


def post_graphql(variables: dict, retries: int, sleep_seconds: float) -> dict:
    body = json.dumps({"query": QUERY, "variables": variables}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "SeiyuuRoleProfiler/0.1 role-cache-builder",
    }
    for attempt in range(retries + 1):
        request = urllib.request.Request(ANILIST_GRAPHQL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            retry_after = error.headers.get("Retry-After")
            if error.code == 429 and retry_after:
                time.sleep(float(retry_after) + 1.0)
            elif attempt < retries:
                time.sleep(sleep_seconds * (2**attempt))
            else:
                raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(sleep_seconds * (2**attempt))
            else:
                raise
    raise RuntimeError("unreachable retry loop")


def fetch_staff_roles(staff: dict, args: argparse.Namespace) -> dict:
    raw_path = args.raw_dir / f"staff_{staff['seiyuu_id']}.json"
    if raw_path.exists() and not args.refresh:
        return read_json(raw_path)

    sort_payloads = {}
    for sort in args.sorts:
        pages = []
        page = 1
        while True:
            data = post_graphql(
                {
                    "id": int(staff["seiyuu_id"]),
                    "page": page,
                    "perPage": args.per_page,
                    "sort": [sort],
                },
                args.retries,
                args.sleep,
            )
            if data.get("errors"):
                raise RuntimeError(json.dumps(data["errors"], ensure_ascii=False))
            staff_payload = data.get("data", {}).get("Staff")
            if not staff_payload:
                break
            connection = staff_payload.get("characterMedia") or {}
            pages.append(connection)
            page_info = connection.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                sort_payloads[sort] = {"staff": staff_payload, "pages": pages}
                break
            page += 1
            time.sleep(args.sleep)

    payload = {
        "fetched_at": utc_now(),
        "staff_seed": staff,
        "sorts": args.sorts,
        "payloads": sort_payloads,
    }
    write_json(raw_path, payload)
    return payload


def iter_edges(raw: dict) -> list[dict]:
    output = []
    seen = set()
    for sort, sort_payload in raw.get("payloads", {}).items():
        staff_payload = sort_payload.get("staff") or {}
        for page in sort_payload.get("pages") or []:
            for edge in page.get("edges") or []:
                media = edge.get("node") or {}
                for character in edge.get("characters") or []:
                    if not character or not media.get("id"):
                        continue
                    key = (int(character["id"]), int(media["id"]), edge.get("characterRole") or "", sort)
                    if key in seen:
                        continue
                    seen.add(key)
                    output.append(
                        {
                            "staff": staff_payload,
                            "character": character,
                            "media": media,
                            "character_role": edge.get("characterRole") or "",
                            "staff_role": edge.get("staffRole") or "",
                            "sort_source": sort,
                        }
                    )
    return output


def build_compact_cache(raw_payloads: list[dict], min_favourites: int) -> dict:
    grouped: dict[tuple[int, int], dict] = {}
    total_edges = 0
    kept_credit_edges = 0
    for raw in raw_payloads:
        seed = raw.get("staff_seed") or {}
        for edge in iter_edges(raw):
            total_edges += 1
            character = edge["character"]
            favourites = int(character.get("favourites") or 0)
            if favourites < min_favourites:
                continue
            kept_credit_edges += 1
            staff_payload = edge.get("staff") or {}
            media = edge["media"]
            seiyuu_id = int(seed.get("seiyuu_id") or staff_payload.get("id"))
            character_id = int(character["id"])
            key = (seiyuu_id, character_id)
            row = grouped.setdefault(
                key,
                {
                    "seiyuu": {
                        "seiyuu_id": seiyuu_id,
                        "name": seed.get("name") or (staff_payload.get("name") or {}).get("full") or "",
                        "native_name": seed.get("native_name") or (staff_payload.get("name") or {}).get("native") or "",
                        "language": staff_payload.get("languageV2") or "Japanese",
                        "image": seed.get("image") or ((staff_payload.get("image") or {}).get("medium") or ""),
                        "site_url": seed.get("site_url") or staff_payload.get("siteUrl") or f"https://anilist.co/staff/{seiyuu_id}",
                    },
                    "character": {
                        "character_id": character_id,
                        "name": (character.get("name") or {}).get("full") or "",
                        "native_name": (character.get("name") or {}).get("native") or "",
                        "gender": character.get("gender") or "",
                        "favourites": favourites,
                        "image": (character.get("image") or {}).get("medium") or "",
                        "site_url": character.get("siteUrl") or f"https://anilist.co/character/{character_id}",
                        "first_anime": "",
                    },
                    "character_role": edge.get("character_role") or "",
                    "anime": [],
                    "first_year": None,
                    "latest_year": None,
                    "credit_count": 0,
                    "splits": ["current_seiyuu_full_anilist"],
                },
            )
            media_year = year(media)
            if media_year is not None:
                row["first_year"] = media_year if row["first_year"] is None else min(row["first_year"], media_year)
                row["latest_year"] = media_year if row["latest_year"] is None else max(row["latest_year"], media_year)
                if not row["character"]["first_anime"] or media_year == row["first_year"]:
                    row["character"]["first_anime"] = title(media)
            row["anime"].append(
                {
                    "anime_id": int(media["id"]),
                    "title": title(media),
                    "year": media_year,
                    "site_url": media.get("siteUrl") or f"https://anilist.co/anime/{media['id']}",
                    "mal_url": f"https://myanimelist.net/anime/{media['idMal']}" if media.get("idMal") else "",
                }
            )
            row["credit_count"] += 1

    seiyuu_counts: dict[int, dict] = defaultdict(lambda: {"role_count": 0, "character_count": 0, "first_year": None})
    for row in grouped.values():
        seiyuu_id = int(row["seiyuu"]["seiyuu_id"])
        seiyuu_counts[seiyuu_id]["role_count"] += int(row["credit_count"])
        seiyuu_counts[seiyuu_id]["character_count"] += 1
        if row["first_year"]:
            current = seiyuu_counts[seiyuu_id]["first_year"]
            seiyuu_counts[seiyuu_id]["first_year"] = row["first_year"] if current is None else min(current, row["first_year"])

    for row in grouped.values():
        counts = seiyuu_counts[int(row["seiyuu"]["seiyuu_id"])]
        row["seiyuu"] = {**row["seiyuu"], **counts}
        deduped_anime = {}
        for anime in row["anime"]:
            deduped_anime[int(anime["anime_id"])] = anime
        row["anime"] = sorted(deduped_anime.values(), key=lambda anime: (anime["year"] or 9999, anime["title"]))[:12]

    return {
        "generated_at": utc_now(),
        "source": "cache_current_seiyuu_full_anilist_roles.py",
        "parameters": {
            "min_favourites": min_favourites,
            "sampling_frame": "current_profiler_seiyuu_full_anilist_character_media",
            "year_floor": None,
        },
        "counts": {
            "raw_credit_edges": total_edges,
            "kept_credit_edges": kept_credit_edges,
            "seiyuu_character_pairs": len(grouped),
            "seiyuu": len(seiyuu_counts),
            "characters": len({key[1] for key in grouped}),
        },
        "roles": sorted(grouped.values(), key=lambda row: (row["seiyuu"]["name"], row["character"]["name"])),
    }


def main() -> None:
    args = parse_args()
    profiles_payload = read_json(args.profiles)
    staff = staff_rows(profiles_payload)
    if args.staff_ids_file:
        wanted_ids = {
            int(line.strip())
            for line in args.staff_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        staff = [row for row in staff if int(row["seiyuu_id"]) in wanted_ids]
    if args.staff_id:
        staff = [row for row in staff if int(row["seiyuu_id"]) == args.staff_id]
    if args.staff_name:
        target = args.staff_name.casefold()
        staff = [row for row in staff if str(row["name"]).casefold() == target]
    if (args.staff_id or args.staff_name) and not staff:
        raise SystemExit("No matching staff member found in profile manifest.")
    if args.limit:
        staff = staff[: args.limit]
    raw_payloads = []
    for index, row in enumerate(staff, start=1):
        raw = fetch_staff_roles(row, args)
        raw_payloads.append(raw)
        edge_count = sum(len(page.get("edges") or []) for payload in raw.get("payloads", {}).values() for page in payload.get("pages") or [])
        print(f"{index}/{len(staff)} {row['name']} edges={edge_count}")
    output = build_compact_cache(raw_payloads, args.min_favourites)
    write_json(args.output, output)
    print(f"wrote {args.output} with {output['counts']}")


if __name__ == "__main__":
    main()
