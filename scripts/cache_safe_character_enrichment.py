#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_APIS = {
    "en": "https://en.wikipedia.org/w/api.php",
    "es": "https://es.wikipedia.org/w/api.php",
    "fr": "https://fr.wikipedia.org/w/api.php",
    "ja": "https://ja.wikipedia.org/w/api.php",
    "zh": "https://zh.wikipedia.org/w/api.php",
}
BANGUMI_CHARACTER_SEARCH = "https://api.bgm.tv/v0/search/characters"
BANGUMI_CHARACTER_SUBJECTS = "https://api.bgm.tv/v0/characters/{id}/subjects"
WIKIPEDIA_CHARACTER_TERMS = {
    "en": ("characters", "list of characters"),
    "es": ("personajes", "lista de personajes"),
    "fr": ("personnages", "liste des personnages"),
    "ja": ("登場人物", "キャラクター"),
    "zh": ("角色", "登場人物"),
}

WIKIDATA_TRAIT_PROPERTY = "P9652"

SOURCE_POLICIES = {
    "wikidata": {
        "license": "CC0",
        "license_url": "https://www.wikidata.org/wiki/Wikidata:Licensing",
        "terms_url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
        "attribution": "Wikidata attribution appreciated; CC0 does not require attribution.",
        "public_use": "allowed",
    },
    "wikipedia": {
        "license": "CC BY-SA / GFDL page-dependent Wikimedia content license",
        "license_url": "https://www.mediawiki.org/wiki/API:Licensing",
        "terms_url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
        "attribution": "Attribute article URL/authors and preserve compatible license for reused/adapted text.",
        "public_use": "allowed_with_attribution_and_share_alike",
    },
    "bangumi": {
        "license": "CC BY-SA for entry/character information per Bangumi copyright page",
        "license_url": "https://bgm.tv/about/copyright",
        "terms_url": "https://bgm.tv/about/copyright",
        "attribution": "Attribute Bangumi source page and avoid raw/bulk redistribution of platform data.",
        "public_use": "allowed_with_attribution_and_platform_limits",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache descriptor-enrichment evidence from source-reviewed public APIs."
    )
    parser.add_argument("--roles-input", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/safe_enrichment"))
    parser.add_argument("--min-favourites", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="0 means all matching characters.")
    parser.add_argument("--sleep-seconds", type=float, default=0.35)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--wikipedia-languages", default="en,es,fr,ja,zh")
    parser.add_argument("--wikidata-limit", type=int, default=5)
    parser.add_argument("--wikipedia-max-extract-chars", type=int, default=3500)
    parser.add_argument("--wikipedia-broad-search-limit", type=int, default=4)
    parser.add_argument("--wikipedia-broad-pages-per-character", type=int, default=5)
    parser.add_argument("--wikipedia-broad-snippet-chars", type=int, default=900)
    parser.add_argument("--include-wikipedia-broad", action="store_true")
    parser.add_argument("--bangumi-limit", type=int, default=5)
    parser.add_argument("--bangumi-min-score", type=float, default=2.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--source", action="append", choices=("wikidata", "wikipedia", "bangumi"), default=[])
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def normalize_name(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"[\u30fb・·•]", " ", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9 ]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def normalize_loose(value: str) -> str:
    value = html.unescape(value or "").lower()
    value = re.sub(r"[\u30fb・·•]", " ", value)
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def normalize_for_search(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def name_keys(value: str) -> set[str]:
    norm = normalize_name(value)
    loose = normalize_loose(value)
    keys = {key for key in (norm, loose) if key}
    if norm:
        parts = norm.split()
        keys.add(" ".join(reversed(parts)))
        keys.add(" ".join(sorted(parts)))
    return keys


def query_names(local: dict) -> list[str]:
    names = []
    for value in (local.get("name") or "", local.get("native_name") or ""):
        if value and value not in names:
            names.append(value)
    norm = normalize_name(local.get("name") or "")
    parts = norm.split()
    if len(parts) >= 2:
        reversed_name = " ".join(reversed(parts))
        if reversed_name and reversed_name not in names:
            names.append(reversed_name)
    return names


def anime_titles(local: dict) -> list[str]:
    titles = []
    for value in [local.get("first_anime") or ""] + [row.get("title") or "" for row in local.get("anime") or []]:
        value = normalize_for_search(value)
        if value and value not in titles:
            titles.append(value)
    return titles[:5]


def fetch_json(url: str, *, method: str = "GET", body: dict | None = None, timeout: float = 30.0) -> dict:
    data = None
    headers = {
        "User-Agent": "SeiyuuRoleProfiler/0.1 safe-source-enrichment contact: local-research",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def raw_path(output_dir: Path, source: str, key: str) -> Path:
    return output_dir / "raw" / source / f"{key}.json"


def cached_json(
    output_dir: Path,
    source: str,
    key: str,
    request: dict,
    args: argparse.Namespace,
) -> dict:
    path = raw_path(output_dir, source, key)
    if path.exists() and not args.refresh:
        return read_json(path)
    if args.offline:
        raise FileNotFoundError(f"missing raw cache {path}")
    try:
        if request["method"] == "POST":
            payload = fetch_json(request["url"], method="POST", body=request["body"])
        else:
            payload = fetch_json(request["url"])
        cached = {"ok": True, "generated_at": utc_now(), "request": request, "response": payload}
    except urllib.error.HTTPError as exc:
        cached = {"ok": False, "generated_at": utc_now(), "request": request, "status": exc.code, "error": str(exc)}
    except Exception as exc:
        cached = {"ok": False, "generated_at": utc_now(), "request": request, "status": None, "error": repr(exc)}
    write_json(path, cached)
    time.sleep(args.sleep_seconds)
    return cached


def load_roles(path: Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("roles"), list):
        return payload["roles"]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported roles payload: {path}")


def unique_characters(args: argparse.Namespace) -> list[dict]:
    roles = load_roles(args.roles_input)
    requested = set().union(*(name_keys(value) for value in args.name)) if args.name else None
    by_id: dict[int, dict] = {}
    for role in roles:
        character = role.get("character") or {}
        favourites = int(character.get("favourites") or 0)
        if favourites < args.min_favourites:
            continue
        if requested and not name_keys(character.get("name") or "").intersection(requested):
            continue
        character_id = int(character["character_id"])
        row = by_id.setdefault(
            character_id,
            {
                "anilist_character_id": character_id,
                "name": character.get("name") or "",
                "native_name": character.get("native_name") or "",
                "first_anime": character.get("first_anime") or "",
                "favourites": favourites,
                "site_url": character.get("site_url") or "",
                "anime": [],
            },
        )
        for anime in role.get("anime") or []:
            slim = {
                "title": anime.get("title") or "",
                "year": anime.get("year"),
                "site_url": anime.get("site_url") or "",
            }
            if slim not in row["anime"]:
                row["anime"].append(slim)
    rows = sorted(by_id.values(), key=lambda item: (-item["favourites"], item["name"]))
    if args.limit:
        rows = rows[: args.limit]
    return rows


def wikidata_search(name: str, args: argparse.Namespace) -> dict:
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "uselang": "en",
        "type": "item",
        "limit": str(args.wikidata_limit),
        "format": "json",
    }
    url = f"{WIKIDATA_API}?{urllib.parse.urlencode(params)}"
    return cached_json(args.output_dir, "wikidata", f"search_{short_hash(name)}", {"method": "GET", "url": url}, args)


def wikidata_entities(ids: list[str], args: argparse.Namespace) -> dict:
    if not ids:
        return {"ok": True, "response": {"entities": {}}}
    params = {
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "labels|descriptions|aliases|claims|sitelinks",
        "languages": "en|es|fr|ja|zh",
        "sitefilter": "enwiki|eswiki|frwiki|jawiki|zhwiki",
        "format": "json",
    }
    url = f"{WIKIDATA_API}?{urllib.parse.urlencode(params)}"
    return cached_json(args.output_dir, "wikidata", f"entities_{short_hash('|'.join(ids))}", {"method": "GET", "url": url}, args)


def wikidata_labels(ids: list[str], args: argparse.Namespace) -> dict:
    if not ids:
        return {}
    params = {
        "action": "wbgetentities",
        "ids": "|".join(sorted(set(ids))),
        "props": "labels",
        "languages": "en",
        "format": "json",
    }
    url = f"{WIKIDATA_API}?{urllib.parse.urlencode(params)}"
    cached = cached_json(args.output_dir, "wikidata", f"labels_{short_hash('|'.join(sorted(set(ids))))}", {"method": "GET", "url": url}, args)
    labels = {}
    for qid, entity in ((cached.get("response") or {}).get("entities") or {}).items():
        label = ((entity.get("labels") or {}).get("en") or {}).get("value")
        if label:
            labels[qid] = label
    return labels


def entity_trait_ids(entity: dict) -> list[str]:
    output = []
    for claim in (entity.get("claims") or {}).get(WIKIDATA_TRAIT_PROPERTY) or []:
        datavalue = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
        numeric = datavalue.get("numeric-id")
        if numeric:
            output.append(f"Q{numeric}")
    return output


def score_wikidata_candidate(local: dict, candidate: dict, entity: dict) -> float:
    score = 0.0
    local_keys = name_keys(local["name"])
    if local.get("native_name"):
        local_keys.update(name_keys(local["native_name"]))
    text_values = [
        candidate.get("label") or "",
        candidate.get("description") or "",
        *(alias.get("value") or "" for lang in (entity.get("aliases") or {}).values() for alias in lang),
        *(((label or {}).get("value") or "") for label in (entity.get("labels") or {}).values()),
    ]
    candidate_keys = set()
    for value in text_values:
        candidate_keys.update(name_keys(value))
    if local_keys.intersection(candidate_keys):
        score += 4.0
    desc = " ".join(text_values).lower()
    if any(term in desc for term in ("fictional", "character", "anime", "manga", "visual novel")):
        score += 1.5
    sitelinks = entity.get("sitelinks") or {}
    if sitelinks:
        score += min(2.0, len(sitelinks) * 0.4)
    if entity_trait_ids(entity):
        score += 2.0
    return score


def wikidata_entity_looks_character(candidate: dict, entity: dict) -> bool:
    values = [
        candidate.get("description") or "",
        *(((description or {}).get("value") or "") for description in (entity.get("descriptions") or {}).values()),
    ]
    text = " ".join(values).lower()
    if any(term in text for term in ("fictional", "character", "anime", "manga", "visual novel", "video game")):
        return True
    return bool(entity_trait_ids(entity))


def collect_wikidata(local: dict, args: argparse.Namespace) -> dict:
    names = query_names(local)
    search_rows = []
    ids = []
    for name in names:
        cached = wikidata_search(name, args)
        for candidate in (cached.get("response") or {}).get("search") or []:
            qid = candidate.get("id")
            if qid and qid not in ids:
                ids.append(qid)
            search_rows.append(candidate)
    entities = (wikidata_entities(ids[: args.wikidata_limit], args).get("response") or {}).get("entities") or {}
    by_id = {row.get("id"): row for row in search_rows if row.get("id")}
    scored = []
    all_trait_ids = []
    for qid, entity in entities.items():
        candidate = by_id.get(qid) or {"id": qid}
        if not wikidata_entity_looks_character(candidate, entity):
            continue
        score = score_wikidata_candidate(local, candidate, entity)
        if score >= 4.0:
            traits = entity_trait_ids(entity)
            all_trait_ids.extend(traits)
            scored.append((score, qid, candidate, entity, traits))
    trait_labels = wikidata_labels(all_trait_ids, args) if all_trait_ids else {}
    matches = []
    for score, qid, candidate, entity, traits in sorted(scored, key=lambda item: (-item[0], item[1]))[:3]:
        matches.append(
            {
                "source": "wikidata",
                "qid": qid,
                "url": f"https://www.wikidata.org/wiki/{qid}",
                "score": round(score, 2),
                "label": candidate.get("label") or ((entity.get("labels") or {}).get("en") or {}).get("value"),
                "description": candidate.get("description") or ((entity.get("descriptions") or {}).get("en") or {}).get("value"),
                "traits": [{"qid": trait, "label": trait_labels.get(trait)} for trait in traits],
                "sitelinks": entity.get("sitelinks") or {},
                "policy": SOURCE_POLICIES["wikidata"],
            }
        )
    return {"matches": matches}


def wikipedia_extract(lang: str, title: str, args: argparse.Namespace) -> dict:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": "1",
        "exintro": "0",
        "inprop": "url",
        "redirects": "1",
        "titles": title,
        "format": "json",
        "formatversion": "2",
    }
    url = f"{WIKIPEDIA_APIS[lang]}?{urllib.parse.urlencode(params)}"
    return cached_json(args.output_dir, "wikipedia", f"{lang}_{short_hash(title)}", {"method": "GET", "url": url}, args)


def wikipedia_search(lang: str, query: str, args: argparse.Namespace) -> dict:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(args.wikipedia_broad_search_limit),
        "format": "json",
        "formatversion": "2",
    }
    url = f"{WIKIPEDIA_APIS[lang]}?{urllib.parse.urlencode(params)}"
    return cached_json(args.output_dir, "wikipedia_search", f"{lang}_{short_hash(query)}", {"method": "GET", "url": url}, args)


def character_needles(local: dict) -> list[str]:
    values = query_names(local)
    # Add name pieces for pages that use shortened names, but avoid one-letter pieces.
    for value in list(values):
        for part in re.split(r"\s+", normalize_for_search(value)):
            if len(part) >= 4 and part not in values:
                values.append(part)
    return values


def text_snippets_for_character(text: str, local: dict, window: int) -> list[dict]:
    if not text:
        return []
    snippets = []
    seen = set()
    lowered = text.lower()
    for needle in character_needles(local):
        if not needle:
            continue
        idx = lowered.find(needle.lower())
        if idx < 0:
            continue
        start = max(0, idx - window // 2)
        end = min(len(text), idx + len(needle) + window // 2)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        key = (needle.lower(), snippet)
        if snippet and key not in seen:
            snippets.append({"needle": needle, "snippet": snippet})
            seen.add(key)
    return snippets[:5]


def title_looks_like_character_page(lang: str, title: str) -> bool:
    title_l = normalize_loose(title)
    terms = WIKIPEDIA_CHARACTER_TERMS.get(lang, ())
    return any(normalize_loose(term) in title_l for term in terms)


def wikipedia_broad_queries(local: dict, lang: str) -> list[str]:
    queries = []
    terms = WIKIPEDIA_CHARACTER_TERMS.get(lang, ("characters",))
    names = query_names(local)[:2]
    for title in anime_titles(local):
        for term in terms:
            for query in (f"{title} {term}", f"{title} {names[0]}" if names else ""):
                query = normalize_for_search(query)
                if query and query not in queries:
                    queries.append(query)
    return queries[:8]


def collect_wikipedia(local: dict, wikidata: dict, args: argparse.Namespace) -> dict:
    languages = {lang.strip() for lang in args.wikipedia_languages.split(",") if lang.strip()}
    rows = []
    seen = set()
    for match in wikidata.get("matches") or []:
        for site_key, sitelink in (match.get("sitelinks") or {}).items():
            lang = {"enwiki": "en", "eswiki": "es", "frwiki": "fr", "jawiki": "ja", "zhwiki": "zh"}.get(site_key)
            if not lang or lang not in languages:
                continue
            title = sitelink.get("title") or ""
            if not title or (lang, title) in seen:
                continue
            seen.add((lang, title))
            cached = wikipedia_extract(lang, title, args)
            for page in ((cached.get("response") or {}).get("query") or {}).get("pages") or []:
                extract = page.get("extract") or ""
                if not extract:
                    continue
                rows.append(
                    {
                        "source": f"{lang}wiki",
                        "language": lang,
                        "title": page.get("title") or title,
                        "url": page.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                        "extract": extract[: args.wikipedia_max_extract_chars],
                        "policy": SOURCE_POLICIES["wikipedia"],
                    }
                )
    return {"matches": rows}


def collect_wikipedia_broad(local: dict, args: argparse.Namespace) -> dict:
    languages = {lang.strip() for lang in args.wikipedia_languages.split(",") if lang.strip()}
    rows = []
    seen_titles = set()
    for lang in languages:
        if lang not in WIKIPEDIA_APIS:
            continue
        for query in wikipedia_broad_queries(local, lang):
            search = wikipedia_search(lang, query, args)
            results = ((search.get("response") or {}).get("query") or {}).get("search") or []
            for result in results:
                title = result.get("title") or ""
                if not title or (lang, title) in seen_titles:
                    continue
                if not title_looks_like_character_page(lang, title) and local["name"].lower() not in (
                    result.get("snippet") or ""
                ).lower():
                    continue
                cached = wikipedia_extract(lang, title, args)
                for page in ((cached.get("response") or {}).get("query") or {}).get("pages") or []:
                    extract = page.get("extract") or ""
                    if not extract:
                        continue
                    mention_snippets = text_snippets_for_character(extract, local, args.wikipedia_broad_snippet_chars)
                    if not mention_snippets and not title_looks_like_character_page(lang, page.get("title") or title):
                        continue
                    rows.append(
                        {
                            "source": f"{lang}wiki_broad",
                            "language": lang,
                            "query": query,
                            "title": page.get("title") or title,
                            "url": page.get("fullurl")
                            or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                            "extract": extract[: args.wikipedia_max_extract_chars],
                            "mention_snippets": mention_snippets,
                            "policy": SOURCE_POLICIES["wikipedia"],
                        }
                    )
                    seen_titles.add((lang, title))
                    if len(rows) >= args.wikipedia_broad_pages_per_character:
                        return {"matches": rows}
    return {"matches": rows}


def bangumi_search(local: dict, query: str, args: argparse.Namespace) -> dict:
    body = {"keyword": query, "limit": args.bangumi_limit}
    key = f"search_{local['anilist_character_id']}_{short_hash(query)}"
    return cached_json(
        args.output_dir,
        "bangumi",
        key,
        {"method": "POST", "url": BANGUMI_CHARACTER_SEARCH, "body": body},
        args,
    )


def bangumi_subjects(candidate: dict, args: argparse.Namespace) -> list[dict]:
    character_id = candidate.get("id")
    if not character_id:
        return []
    url = BANGUMI_CHARACTER_SUBJECTS.format(id=character_id)
    cached = cached_json(
        args.output_dir,
        "bangumi_subjects",
        f"subjects_{character_id}",
        {"method": "GET", "url": url},
        args,
    )
    response = cached.get("response") or []
    return response if isinstance(response, list) else []


def title_tokens(value: str) -> set[str]:
    tokens = set(normalize_name(value).split())
    # Keep a loose CJK/Japanese string as a token-like key when ASCII drops it.
    loose = normalize_loose(value)
    if loose and not tokens:
        tokens.add(loose)
    return {token for token in tokens if len(token) >= 3}


def local_anime_title_tokens(local: dict) -> set[str]:
    tokens = set()
    for value in [local.get("first_anime") or ""] + [row.get("title") or "" for row in local.get("anime") or []]:
        tokens.update(title_tokens(value))
    return tokens


def subject_title_tokens(subjects: list[dict]) -> set[str]:
    tokens = set()
    for subject in subjects:
        tokens.update(title_tokens(subject.get("name") or ""))
        tokens.update(title_tokens(subject.get("name_cn") or ""))
    return tokens


def bangumi_candidate_score(local: dict, candidate: dict, subjects: list[dict]) -> float:
    local_ascii = {normalize_name(local.get("name") or "")}
    local_norm = normalize_name(local.get("name") or "")
    if local_norm:
        parts = local_norm.split()
        if len(parts) >= 2:
            local_ascii.add(" ".join(reversed(parts)))
    local_loose = {normalize_loose(local.get("name") or "")}
    if local.get("native_name"):
        local_loose.add(normalize_loose(local["native_name"]))
    values = [candidate.get("name") or "", candidate.get("name_cn") or ""]
    for item in candidate.get("infobox") or []:
        if item.get("key") in {"别名", "簡体中文名", "日文名", "英文名"}:
            value = item.get("value")
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        values.append(str(row.get("v") or ""))
                    else:
                        values.append(str(row))
    candidate_ascii = set()
    candidate_loose = set()
    for value in values:
        ascii_value = normalize_name(value)
        loose_value = normalize_loose(value)
        if ascii_value:
            candidate_ascii.add(ascii_value)
        if loose_value:
            candidate_loose.add(loose_value)
    score = 0.0
    exact_ascii = bool(local_ascii.intersection(candidate_ascii))
    exact_loose = bool(local_loose.intersection(candidate_loose))
    exact_native = bool(local.get("native_name") and normalize_loose(local["native_name"]) in candidate_loose)
    title_overlap = len(local_anime_title_tokens(local).intersection(subject_title_tokens(subjects)))
    local_name_token_count = len(normalize_name(local.get("name") or "").split())
    if not (exact_ascii or exact_loose):
        return 0.0
    if not exact_native and local_name_token_count < 2 and title_overlap == 0:
        return 0.0
    score += 4.0
    if exact_native:
        score += 1.0
    score += min(2.0, title_overlap * 0.75)
    if candidate.get("summary"):
        score += 1.0
    if candidate.get("images"):
        score += 0.25
    return score


def collect_bangumi(local: dict, args: argparse.Namespace) -> dict:
    queries = query_names(local)
    scored = []
    for query in queries:
        cached = bangumi_search(local, query, args)
        candidates = (cached.get("response") or {}).get("data") or []
        for candidate in candidates:
            subjects = bangumi_subjects(candidate, args)
            score = bangumi_candidate_score(local, candidate, subjects)
            if score >= args.bangumi_min_score:
                scored.append((score, candidate, subjects))
    matches = []
    seen = set()
    for score, candidate, subjects in sorted(scored, key=lambda item: (-item[0], item[1].get("id") or 0))[:3]:
        cid = candidate.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        matches.append(
            {
                "source": "bangumi",
                "bangumi_character_id": cid,
                "url": f"https://bgm.tv/character/{cid}" if cid else "",
                "score": round(score, 2),
                "name": candidate.get("name") or "",
                "name_cn": candidate.get("name_cn") or "",
                "summary": candidate.get("summary") or "",
                "infobox": candidate.get("infobox") or [],
                "subjects": [
                    {
                        "id": subject.get("id"),
                        "type": subject.get("type"),
                        "name": subject.get("name") or "",
                        "name_cn": subject.get("name_cn") or "",
                        "staff": subject.get("staff") or "",
                    }
                    for subject in subjects[:12]
                ],
                "policy": SOURCE_POLICIES["bangumi"],
            }
        )
    return {"matches": matches}


def processed_ids(path: Path) -> set[int]:
    ids = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                ids.add(int(json.loads(line)["anilist_character_id"]))
            except Exception:
                continue
    return ids


def build_row(local: dict, args: argparse.Namespace) -> dict:
    active_sources = set(args.source or ("wikidata", "wikipedia", "bangumi"))
    row = {
        "generated_at": utc_now(),
        "anilist_character_id": local["anilist_character_id"],
        "name": local["name"],
        "native_name": local["native_name"],
        "first_anime": local["first_anime"],
        "favourites": local["favourites"],
        "site_url": local["site_url"],
        "source_policies": {key: SOURCE_POLICIES[key] for key in sorted(active_sources)},
        "wikidata": {"matches": []},
        "wikipedia": {"matches": []},
        "wikipedia_broad": {"matches": []},
        "bangumi": {"matches": []},
        "coverage": {},
    }
    if "wikidata" in active_sources or "wikipedia" in active_sources:
        row["wikidata"] = collect_wikidata(local, args)
    if "wikipedia" in active_sources:
        row["wikipedia"] = collect_wikipedia(local, row["wikidata"], args)
        if args.include_wikipedia_broad:
            row["wikipedia_broad"] = collect_wikipedia_broad(local, args)
    if "bangumi" in active_sources:
        row["bangumi"] = collect_bangumi(local, args)
    row["coverage"] = {
        "wikidata_matches": len(row["wikidata"]["matches"]),
        "wikidata_trait_count": sum(len(match.get("traits") or []) for match in row["wikidata"]["matches"]),
        "wikipedia_extract_count": len(row["wikipedia"]["matches"]),
        "wikipedia_broad_page_count": len(row["wikipedia_broad"]["matches"]),
        "wikipedia_broad_snippet_count": sum(
            len(match.get("mention_snippets") or []) for match in row["wikipedia_broad"]["matches"]
        ),
        "bangumi_matches": len(row["bangumi"]["matches"]),
        "bangumi_summary_count": sum(1 for match in row["bangumi"]["matches"] if match.get("summary")),
    }
    return row


def summarize(output_jsonl: Path, output_dir: Path) -> None:
    rows = []
    if output_jsonl.exists():
        with output_jsonl.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    n = len(rows)
    summary = {
        "generated_at": utc_now(),
        "rows": n,
        "coverage": {
            "wikidata_rows": sum(row["coverage"]["wikidata_matches"] > 0 for row in rows),
            "wikidata_trait_rows": sum(row["coverage"]["wikidata_trait_count"] > 0 for row in rows),
            "wikipedia_rows": sum(row["coverage"]["wikipedia_extract_count"] > 0 for row in rows),
            "wikipedia_broad_rows": sum(row["coverage"].get("wikipedia_broad_page_count", 0) > 0 for row in rows),
            "wikipedia_broad_snippet_rows": sum(
                row["coverage"].get("wikipedia_broad_snippet_count", 0) > 0 for row in rows
            ),
            "bangumi_rows": sum(row["coverage"]["bangumi_matches"] > 0 for row in rows),
            "bangumi_summary_rows": sum(row["coverage"]["bangumi_summary_count"] > 0 for row in rows),
        },
        "source_policies": SOURCE_POLICIES,
    }
    write_json(output_dir / "summary.json", summary)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "source_policies.json", {"generated_at": utc_now(), "sources": SOURCE_POLICIES})

    output_jsonl = args.output_dir / "character_safe_enrichment.jsonl"
    done = processed_ids(output_jsonl) if not args.refresh else set()
    if args.refresh and output_jsonl.exists():
        output_jsonl.unlink()

    characters = unique_characters(args)
    print(
        json.dumps(
            {"requested": len(characters), "already_done": len(done), "sources": args.source or ["wikidata", "wikipedia", "bangumi"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    processed = 0
    for index, local in enumerate(characters, start=1):
        if local["anilist_character_id"] in done:
            continue
        row = build_row(local, args)
        append_jsonl(output_jsonl, row)
        processed += 1
        print(
            json.dumps(
                {
                    "index": index,
                    "id": local["anilist_character_id"],
                    "name": local["name"],
                    "coverage": row["coverage"],
                    "processed": processed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if processed % args.checkpoint_every == 0:
            summarize(output_jsonl, args.output_dir)
    summarize(output_jsonl, args.output_dir)


if __name__ == "__main__":
    main()
