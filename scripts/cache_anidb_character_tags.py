#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANIME_MAP_URL = "https://raw.githubusercontent.com/Fribb/anime-lists/master/anime-list-full.json"
ANIDB_HTTP_API = "http://api.anidb.net:9001/httpapi"
ANIDB_CHARACTER_URL = "https://anidb.net/character/{character_id}"
KOMETA_ANIDB_MIRROR = "https://utilities.kometa.wiki/anidb-service/anime/{aid}"
ANIDB_SCHEMA_CATEGORIES = {
    "abilities",
    "accessories",
    "age range",
    "clothing",
    "disability",
    "entity",
    "fashion accessories",
    "fetish appeals",
    "habits",
    "looks",
    "maintenance tags",
    "nationality",
    "personality",
    "role",
    "supernatural abilities",
    "traits",
    "weapons",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache AniDB anime XML, crosswalk the local AniList role corpus to AniDB "
            "character ids, and optionally parse cached/fetched AniDB character tag pages."
        )
    )
    parser.add_argument("--roles-input", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/anidb"))
    parser.add_argument("--seiyuu", default="", help="Optional exact/flexible seiyuu-name filter for trial runs.")
    parser.add_argument("--min-character-favourites", type=int, default=100)
    parser.add_argument("--client", default=os.environ.get("ANIDB_CLIENT"))
    parser.add_argument("--clientver", type=int, default=int(os.environ.get("ANIDB_CLIENTVER", "0") or 0))
    parser.add_argument("--protover", type=int, default=1)
    parser.add_argument("--anime-xml-source", choices=["official", "kometa"], default="kometa")
    parser.add_argument("--anime-xml-base-url", default=KOMETA_ANIDB_MIRROR)
    parser.add_argument("--anime-map-url", default=ANIME_MAP_URL)
    parser.add_argument("--refresh-anime-map", action="store_true")
    parser.add_argument("--fetch-anime-xml", action="store_true")
    parser.add_argument("--anime-fetch-jobs", type=int, default=1)
    parser.add_argument("--max-anime-fetches", type=int, default=None)
    parser.add_argument("--request-delay-seconds", type=float, default=4.0)
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


def fetch_bytes(url: str, *, timeout: int = 45) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SeiyuuRoleProfiler/0.1 (+local reproducible research cache)",
            "Accept": "application/json,application/xml,text/html;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def normalize_name(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def name_keys(value: str) -> set[str]:
    norm = normalize_name(value)
    if not norm:
        return set()
    parts = norm.split()
    keys = {norm, " ".join(reversed(parts)), " ".join(sorted(parts))}
    if len(parts) > 2:
        keys.add(" ".join(parts[:2]))
        keys.add(" ".join(sorted(parts[:2])))
    return {key for key in keys if key}


def parse_mal_id(url: str | None) -> int | None:
    if not url:
        return None
    match = re.search(r"/anime/(\d+)", url)
    return int(match.group(1)) if match else None


def load_or_fetch_anime_map(output_dir: Path, source_url: str, refresh: bool) -> list[dict]:
    path = output_dir / "anime-list-full.json"
    if path.exists() and not refresh:
        return read_json(path)
    payload = json.loads(fetch_bytes(source_url).decode("utf-8"))
    write_json(path, payload)
    return payload


def build_anime_lookup(rows: list[dict]) -> tuple[dict[int, int], dict[int, int]]:
    by_anilist: dict[int, int] = {}
    by_mal: dict[int, int] = {}
    for row in rows:
        anidb_id = row.get("anidb_id")
        if not anidb_id:
            continue
        if row.get("anilist_id"):
            by_anilist[int(row["anilist_id"])] = int(anidb_id)
        if row.get("mal_id"):
            by_mal[int(row["mal_id"])] = int(anidb_id)
    return by_anilist, by_mal


def target_roles(roles_payload: dict, min_favourites: int, seiyuu_filter: str = "") -> list[dict]:
    output = []
    seen = set()
    filter_keys = name_keys(seiyuu_filter) if seiyuu_filter else set()
    for role in roles_payload.get("roles", []):
        character = role.get("character", {})
        if int(character.get("favourites") or 0) < min_favourites:
            continue
        if filter_keys and not filter_keys.intersection(name_keys(role.get("seiyuu", {}).get("name") or "")):
            continue
        key = (role.get("seiyuu", {}).get("seiyuu_id"), character.get("character_id"))
        if key in seen:
            continue
        seen.add(key)
        output.append(role)
    return output


def role_anidb_aids(role: dict, by_anilist: dict[int, int], by_mal: dict[int, int]) -> list[int]:
    aids = []
    for anime in role.get("anime", []):
        aid = None
        anilist_id = anime.get("anime_id")
        if anilist_id is not None:
            aid = by_anilist.get(int(anilist_id))
        if aid is None:
            mal_id = parse_mal_id(anime.get("mal_url"))
            if mal_id is not None:
                aid = by_mal.get(mal_id)
        if aid:
            aids.append(aid)
    return sorted(set(aids))


def anidb_xml_path(output_dir: Path, aid: int) -> Path:
    return output_dir / "anime_xml" / f"{aid}.xml"


def fetch_anime_xml(output_dir: Path, aid: int, args: argparse.Namespace) -> bool:
    path = anidb_xml_path(output_dir, aid)
    if path.exists():
        return False
    if args.dry_run:
        return False
    if args.anime_xml_source == "kometa":
        xml_bytes = fetch_bytes(args.anime_xml_base_url.format(aid=aid), timeout=60)
    else:
        if not args.client or not args.clientver:
            raise RuntimeError(
                "Official AniDB HTTP anime XML fetch requires --client and --clientver, or ANIDB_CLIENT/ANIDB_CLIENTVER."
            )
        params = urllib.parse.urlencode(
            {
                "request": "anime",
                "client": args.client,
                "clientver": args.clientver,
                "protover": args.protover,
                "aid": aid,
            }
        )
        xml_bytes = fetch_bytes(f"{ANIDB_HTTP_API}?{params}", timeout=60)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(xml_bytes)
    return True


def parse_anime_xml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError:
        return []
    output = []
    for node in root.findall(".//characters/character"):
        character_id = node.attrib.get("id")
        if not character_id:
            continue
        seiyuu = []
        for seiyuu_node in node.findall("seiyuu"):
            seiyuu.append(
                {
                    "anidb_seiyuu_id": int(seiyuu_node.attrib["id"]) if seiyuu_node.attrib.get("id") else None,
                    "name": (seiyuu_node.text or "").strip(),
                    "picture": seiyuu_node.attrib.get("picture") or "",
                }
            )
        rating = node.find("rating")
        output.append(
            {
                "anidb_character_id": int(character_id),
                "name": (node.findtext("name") or "").strip(),
                "gender": (node.findtext("gender") or "").strip(),
                "type": node.attrib.get("type") or "",
                "picture": (node.findtext("picture") or "").strip(),
                "rating": float(rating.text) if rating is not None and rating.text else None,
                "rating_votes": int(rating.attrib.get("votes", "0")) if rating is not None else 0,
                "seiyuu": seiyuu,
            }
        )
    return output


def parse_anime_tag_taxonomy(output_dir: Path) -> dict:
    tags: dict[str, dict] = {}
    children_by_parent: dict[str, set[str]] = defaultdict(set)
    anime_counts: dict[str, int] = defaultdict(int)
    for path in sorted((output_dir / "anime_xml").glob("*.xml")):
        try:
            root = ET.fromstring(path.read_bytes())
        except ET.ParseError:
            continue
        seen_in_anime = set()
        for node in root.findall(".//tags/tag"):
            tag_id = node.attrib.get("id")
            name = (node.findtext("name") or "").strip()
            if not tag_id or not name:
                continue
            parent_id = node.attrib.get("parentid") or ""
            tags[tag_id] = {
                "tag_id": int(tag_id),
                "name": name,
                "parent_id": int(parent_id) if parent_id else None,
            }
            if parent_id:
                children_by_parent[parent_id].add(tag_id)
            seen_in_anime.add(tag_id)
        for tag_id in seen_in_anime:
            anime_counts[tag_id] += 1

    values_by_parent_name: dict[str, list[str]] = defaultdict(list)
    tag_records_by_parent_name: dict[str, list[dict]] = defaultdict(list)
    for parent_id, child_ids in children_by_parent.items():
        parent = tags.get(parent_id)
        if not parent:
            continue
        parent_name = parent["name"]
        for child_id in sorted(child_ids, key=lambda value: tags.get(value, {}).get("name", "")):
            child = tags.get(child_id)
            if not child:
                continue
            values_by_parent_name[parent_name].append(child["name"])
            tag_records_by_parent_name[parent_name].append(
                {
                    "tag_id": child["tag_id"],
                    "name": child["name"],
                    "parent_id": child["parent_id"],
                    "anime_count": anime_counts.get(child_id, 0),
                }
            )

    character_candidate_parent = "TO BE MOVED TO CHARACTER"
    return {
        "source": "cached AniDB anime XML tag tree",
        "tag_count": len(tags),
        "category_count": len(values_by_parent_name),
        "values_by_category": {key: sorted(set(value)) for key, value in sorted(values_by_parent_name.items())},
        "tag_records_by_category": {key: value for key, value in sorted(tag_records_by_parent_name.items())},
        "character_candidate_category": character_candidate_parent,
        "character_candidate_values": sorted(set(values_by_parent_name.get(character_candidate_parent, []))),
    }


def match_role_to_anidb_character(role: dict, candidates_by_aid: dict[int, list[dict]], aids: list[int]) -> dict | None:
    character = role.get("character", {})
    seiyuu = role.get("seiyuu", {})
    character_keys = name_keys(character.get("name") or "")
    seiyuu_keys = name_keys(seiyuu.get("name") or "")
    matches = []
    for aid in aids:
        for candidate in candidates_by_aid.get(aid, []):
            candidate_keys = name_keys(candidate.get("name") or "")
            if not character_keys.intersection(candidate_keys):
                continue
            seiyuu_match = any(
                seiyuu_keys.intersection(name_keys(row.get("name") or ""))
                for row in candidate.get("seiyuu", [])
            )
            matches.append((aid, candidate, seiyuu_match))
    if not matches:
        return None
    seiyuu_matches = [row for row in matches if row[2]]
    chosen = seiyuu_matches[0] if seiyuu_matches else matches[0]
    aid, candidate, seiyuu_match = chosen
    return {
        "anidb_anime_id": aid,
        "anidb_character_id": candidate["anidb_character_id"],
        "anidb_character_name": candidate["name"],
        "anidb_character_url": ANIDB_CHARACTER_URL.format(character_id=candidate["anidb_character_id"]),
        "matched_by": "name+seiyuu+anime" if seiyuu_match else "name+anime",
        "candidate_count": len(matches),
        "anidb_role_type": candidate.get("type") or "",
        "anidb_rating": candidate.get("rating"),
        "anidb_rating_votes": candidate.get("rating_votes") or 0,
    }


def cached_character_page_path(output_dir: Path, character_id: int) -> Path:
    return output_dir / "character_html" / f"{character_id}.html.gz"


def read_cached_character_html(output_dir: Path, character_id: int) -> str | None:
    path = cached_character_page_path(output_dir, character_id)
    if not path.exists():
        return None
    return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")


def parse_character_tags_from_html(page_html: str) -> list[dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:
        raise RuntimeError("Install beautifulsoup4 to parse cached AniDB character pages.") from error

    soup = BeautifulSoup(page_html, "html.parser")
    output = []
    seen = set()
    for row in soup.select("tr"):
        category_node = row.select_one("th.field")
        value_node = row.select_one("td.value")
        if category_node is None or value_node is None:
            continue
        category = re.sub(r"\s+", " ", category_node.get_text(" ", strip=True)).strip().lower()
        if category not in ANIDB_SCHEMA_CATEGORIES:
            continue
        for anchor in value_node.select('a[href*="/tag/"]'):
            name_node = anchor.select_one(".tagname")
            name = re.sub(
                r"\s+",
                " ",
                (name_node.get_text(" ", strip=True) if name_node else anchor.get_text(" ", strip=True)),
            ).strip()
            if not name:
                continue
            tag_id_match = re.search(r"/tag/(\d+)", str(anchor.get("href") or ""))
            tag_id = int(tag_id_match.group(1)) if tag_id_match else None
            key = (category, tag_id, name.lower())
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "category": category,
                    "tag_id": tag_id,
                    "name": name,
                    "url": urllib.parse.urljoin("https://anidb.net", str(anchor.get("href") or "")),
                }
            )
    if output:
        return output

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "")
        if "/tag/" not in href:
            continue
        name = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
        if not name:
            continue
        tag_id_match = re.search(r"/tag/(\d+)", href)
        tag_id = int(tag_id_match.group(1)) if tag_id_match else None
        category = nearest_schema_category(anchor)
        if not category:
            continue
        key = (category, tag_id, name.lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "category": category,
                "tag_id": tag_id,
                "name": name,
                "url": urllib.parse.urljoin("https://anidb.net", href),
            }
        )
    return output


def stable_tag_key(tag: dict) -> str:
    category = re.sub(r"\s+", "_", str(tag.get("category") or "").strip().lower())
    name = re.sub(r"\s+", "_", str(tag.get("name") or "").strip().lower())
    name = re.sub(r"[^a-z0-9_]+", "", name)
    return f"{category}:{name}" if category and name else name


def nearest_schema_category(anchor: Any) -> str:
    for parent in anchor.parents:
        text = re.sub(r"\s+", " ", parent.get_text(" ", strip=True)).lower()
        hits = [category for category in ANIDB_SCHEMA_CATEGORIES if re.search(rf"\b{re.escape(category)}\b", text)]
        if hits:
            return sorted(hits, key=len, reverse=True)[0]
        if getattr(parent, "name", None) in {"table", "section", "body"}:
            break
    return ""


def build_payload(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    roles_payload = read_json(args.roles_input)
    anime_map = load_or_fetch_anime_map(output_dir, args.anime_map_url, args.refresh_anime_map)
    by_anilist, by_mal = build_anime_lookup(anime_map)
    roles = target_roles(roles_payload, args.min_character_favourites, args.seiyuu)

    aids_by_role = {}
    target_aids = set()
    for index, role in enumerate(roles):
        aids = role_anidb_aids(role, by_anilist, by_mal)
        aids_by_role[index] = aids
        target_aids.update(aids)

    fetched_anime = 0
    if args.fetch_anime_xml:
        aids_to_fetch = [aid for aid in sorted(target_aids) if not anidb_xml_path(output_dir, aid).exists()]
        if args.max_anime_fetches is not None:
            aids_to_fetch = aids_to_fetch[: args.max_anime_fetches]
        if args.anime_xml_source == "official" and args.anime_fetch_jobs != 1:
            raise RuntimeError("Use --anime-fetch-jobs 1 with the official AniDB HTTP API.")
        if args.anime_fetch_jobs <= 1:
            for aid in aids_to_fetch:
                if fetch_anime_xml(output_dir, aid, args):
                    fetched_anime += 1
                    time.sleep(args.request_delay_seconds)
        else:
            with ThreadPoolExecutor(max_workers=args.anime_fetch_jobs) as executor:
                futures = {executor.submit(fetch_anime_xml, output_dir, aid, args): aid for aid in aids_to_fetch}
                for index, future in enumerate(as_completed(futures), start=1):
                    aid = futures[future]
                    try:
                        if future.result():
                            fetched_anime += 1
                    except Exception as error:
                        print(f"warning: failed to fetch AniDB anime {aid}: {error}")
                    if index % 100 == 0:
                        print(f"fetched {fetched_anime}/{len(aids_to_fetch)} anime XML files")

    candidates_by_aid = {
        aid: parse_anime_xml(anidb_xml_path(output_dir, aid))
        for aid in sorted(target_aids)
        if anidb_xml_path(output_dir, aid).exists()
    }

    crosswalk = []
    matched_character_ids = set()
    for index, role in enumerate(roles):
        match = match_role_to_anidb_character(role, candidates_by_aid, aids_by_role[index])
        row = {
            "anilist_character_id": role["character"]["character_id"],
            "anilist_character_name": role["character"]["name"],
            "anilist_character_url": role["character"].get("site_url") or "",
            "seiyuu_id": role["seiyuu"]["seiyuu_id"],
            "seiyuu_name": role["seiyuu"]["name"],
            "anidb_anime_ids": aids_by_role[index],
            "match": match,
        }
        if match:
            matched_character_ids.add(match["anidb_character_id"])
        crosswalk.append(row)

    character_tags = {}
    parse_errors = {}
    for character_id in sorted(matched_character_ids):
        page = read_cached_character_html(output_dir, character_id)
        if not page:
            continue
        try:
            character_tags[str(character_id)] = parse_character_tags_from_html(page)
        except Exception as error:
            parse_errors[str(character_id)] = str(error)

    crosswalk_by_anidb_character_id = {
        str(row["match"]["anidb_character_id"]): row
        for row in crosswalk
        if row.get("match")
    }
    tag_counts = defaultdict(int)
    category_counts = defaultdict(int)
    tag_union = {}
    tags_by_category = defaultdict(list)
    values_by_category = defaultdict(set)
    for character_id, tags in character_tags.items():
        source_row = crosswalk_by_anidb_character_id.get(str(character_id), {})
        character_ref = {
            "anidb_character_id": int(character_id),
            "anilist_character_id": source_row.get("anilist_character_id"),
            "name": source_row.get("anilist_character_name") or source_row.get("match", {}).get("anidb_character_name") or "",
            "seiyuu_name": source_row.get("seiyuu_name") or "",
        }
        for tag in tags:
            tag_counts[tag["name"]] += 1
            category_counts[tag["category"]] += 1
            key = stable_tag_key(tag)
            row = tag_union.setdefault(
                key,
                {
                    "key": key,
                    "category": tag["category"],
                    "name": tag["name"],
                    "tag_id": tag.get("tag_id"),
                    "url": tag.get("url") or "",
                    "character_count": 0,
                    "characters": [],
                },
            )
            row["character_count"] += 1
            row["characters"].append(character_ref)
            values_by_category[tag["category"]].add(tag["name"])

    for key, row in sorted(tag_union.items(), key=lambda item: (item[1]["category"], item[1]["name"])):
        tags_by_category[row["category"]].append(key)
    anime_tag_taxonomy = parse_anime_tag_taxonomy(output_dir)
    character_page_values_by_category = {
        key: sorted(value) for key, value in sorted(values_by_category.items())
    }
    combined_values_by_category = defaultdict(set)
    for category, values in anime_tag_taxonomy["values_by_category"].items():
        combined_values_by_category[category].update(values)
    for category, values in character_page_values_by_category.items():
        combined_values_by_category[category].update(values)

    return {
        "generated_at": utc_now(),
        "source": "cache_anidb_character_tags.py",
        "parameters": {
            "roles_input": str(args.roles_input),
            "seiyuu": args.seiyuu,
            "min_character_favourites": args.min_character_favourites,
            "fetch_anime_xml": args.fetch_anime_xml,
            "anime_xml_source": args.anime_xml_source,
            "anime_map_url": args.anime_map_url,
        },
        "counts": {
            "target_roles": len(roles),
            "target_anidb_anime_ids": len(target_aids),
            "cached_anime_xml": len(candidates_by_aid),
            "fetched_anime_xml": fetched_anime,
            "crosswalk_matches": sum(1 for row in crosswalk if row["match"]),
            "matched_anidb_characters": len(matched_character_ids),
            "cached_character_tag_pages": len(character_tags),
            "fetched_character_pages": 0,
            "characters_with_parsed_tags": len(character_tags),
            "parse_errors": len(parse_errors),
        },
        "notes": [
            "AniDB public HTTP API is anime-centric and provides cast/character ids, not character tags.",
            "Character tag extraction only parses local character_html files if they are provided from a permitted source; this runner does not fetch AniDB character HTML.",
            "AniDB policy disallows scraping HTML pages, so HTML crawling is intentionally excluded from the reproducible pipeline.",
        ],
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "category_counts": dict(sorted(category_counts.items())),
        "tag_union": dict(sorted(tag_union.items())),
        "tags_by_category": {key: sorted(value) for key, value in sorted(tags_by_category.items())},
        "values_by_category": {key: sorted(value) for key, value in sorted(combined_values_by_category.items())},
        "character_page_values_by_category": character_page_values_by_category,
        "anime_tag_taxonomy": anime_tag_taxonomy,
        "anime_tag_values_by_category": anime_tag_taxonomy["values_by_category"],
        "anidb_character_candidate_values": anime_tag_taxonomy["character_candidate_values"],
        "crosswalk": crosswalk,
        "character_tags": character_tags,
        "parse_errors": parse_errors,
    }


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    output = args.output_dir / "anidb_character_tag_cache.json"
    write_json(output, payload)
    write_json(args.output_dir / "anidb_character_crosswalk.json", {"generated_at": payload["generated_at"], "crosswalk": payload["crosswalk"]})
    print(f"wrote {output}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
