#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

from cache_safe_character_enrichment import (
    SOURCE_POLICIES,
    WIKIPEDIA_APIS,
    WIKIPEDIA_CHARACTER_TERMS,
    cached_json,
    normalize_for_search,
    normalize_loose,
    read_json,
    utc_now,
    wikipedia_extract,
    wikipedia_search,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Wikimedia anime-level character/list pages for source-safe descriptor enrichment."
    )
    parser.add_argument("--roles-input", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/wikimedia_anime_character_pages"))
    parser.add_argument("--min-favourites", type=int, default=100)
    parser.add_argument("--limit-anime", type=int, default=0, help="0 means all anime titles.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--languages", default="en,es,fr,ja,zh")
    parser.add_argument("--search-limit", type=int, default=5)
    parser.add_argument("--max-extract-chars", type=int, default=8000)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def load_roles(path: Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("roles"), list):
        return payload["roles"]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported roles payload: {path}")


def clean_title(title: str) -> str:
    title = normalize_for_search(title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def anime_title_rows(args: argparse.Namespace) -> list[dict]:
    roles = load_roles(args.roles_input)
    counts: Counter[str] = Counter()
    favs: defaultdict[str, int] = defaultdict(int)
    examples: defaultdict[str, set[str]] = defaultdict(set)
    for role in roles:
        character = role.get("character") or {}
        if int(character.get("favourites") or 0) < args.min_favourites:
            continue
        anime_rows = role.get("anime") or []
        if not anime_rows and character.get("first_anime"):
            anime_rows = [{"title": character["first_anime"]}]
        for anime in anime_rows:
            title = clean_title(anime.get("title") or "")
            if not title:
                continue
            counts[title] += 1
            favs[title] += int(character.get("favourites") or 0)
            examples[title].add(character.get("name") or "")
    rows = [
        {
            "anime_title": title,
            "character_rows": count,
            "character_favourites": favs[title],
            "example_characters": sorted(name for name in examples[title] if name)[:12],
        }
        for title, count in counts.items()
    ]
    rows.sort(key=lambda row: (-row["character_favourites"], -row["character_rows"], row["anime_title"]))
    if args.limit_anime:
        rows = rows[: args.limit_anime]
    return rows


def title_looks_relevant(lang: str, title: str, snippet: str) -> bool:
    haystack = f"{title} {snippet}"
    haystack_norm = normalize_loose(haystack)
    for term in WIKIPEDIA_CHARACTER_TERMS.get(lang, ()):
        if normalize_loose(term) in haystack_norm:
            return True
    return False


def search_queries(anime_title: str, lang: str) -> list[str]:
    terms = WIKIPEDIA_CHARACTER_TERMS.get(lang, ("characters",))
    queries = []
    for term in terms:
        for query in (f"{anime_title} {term}", f"{term} {anime_title}"):
            query = normalize_for_search(query)
            if query not in queries:
                queries.append(query)
    return queries


def collect_pages(row: dict, args: argparse.Namespace) -> list[dict]:
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    pages = []
    seen = set()
    for lang in languages:
        if lang not in WIKIPEDIA_APIS:
            continue
        for query in search_queries(row["anime_title"], lang):
            # Reuse the cache helper from the character runner, but override the
            # search limit for this anime-title crawler.
            old_limit = args.wikipedia_broad_search_limit if hasattr(args, "wikipedia_broad_search_limit") else None
            args.wikipedia_broad_search_limit = args.search_limit
            search = wikipedia_search(lang, query, args)
            if old_limit is not None:
                args.wikipedia_broad_search_limit = old_limit
            for result in ((search.get("response") or {}).get("query") or {}).get("search") or []:
                title = result.get("title") or ""
                snippet = result.get("snippet") or ""
                if not title or (lang, title) in seen:
                    continue
                if not title_looks_relevant(lang, title, snippet):
                    continue
                extract = wikipedia_extract(lang, title, args)
                for page in ((extract.get("response") or {}).get("query") or {}).get("pages") or []:
                    page_extract = page.get("extract") or ""
                    if not page_extract:
                        continue
                    pages.append(
                        {
                            "source": f"{lang}wiki",
                            "language": lang,
                            "query": query,
                            "title": page.get("title") or title,
                            "url": page.get("fullurl")
                            or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                            "extract": page_extract[: args.max_extract_chars],
                            "policy": SOURCE_POLICIES["wikipedia"],
                        }
                    )
                    seen.add((lang, title))
    return pages


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def processed_titles(path: Path) -> set[str]:
    titles = set()
    if not path.exists():
        return titles
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    titles.add(json.loads(line)["anime_title"])
                except Exception:
                    pass
    return titles


def summarize(output_dir: Path, output_jsonl: Path) -> None:
    rows = []
    if output_jsonl.exists():
        rows = [json.loads(line) for line in output_jsonl.open(encoding="utf-8") if line.strip()]
    write_json(
        output_dir / "summary.json",
        {
            "generated_at": utc_now(),
            "anime_rows": len(rows),
            "with_pages": sum(bool(row.get("pages")) for row in rows),
            "page_count": sum(len(row.get("pages") or []) for row in rows),
            "source_policy": SOURCE_POLICIES["wikipedia"],
        },
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.wikipedia_broad_search_limit = args.search_limit
    args.wikipedia_max_extract_chars = args.max_extract_chars
    output_jsonl = args.output_dir / "anime_character_pages.jsonl"
    done = processed_titles(output_jsonl) if not args.refresh else set()
    if args.refresh and output_jsonl.exists():
        output_jsonl.unlink()
    rows = anime_title_rows(args)
    print(json.dumps({"anime_requested": len(rows), "already_done": len(done), "languages": args.languages}, ensure_ascii=False), flush=True)
    processed = 0
    for index, anime in enumerate(rows, start=1):
        if anime["anime_title"] in done:
            continue
        pages = collect_pages(anime, args)
        payload = {"generated_at": utc_now(), **anime, "pages": pages, "coverage": {"page_count": len(pages)}}
        append_jsonl(output_jsonl, payload)
        processed += 1
        print(
            json.dumps(
                {"index": index, "anime_title": anime["anime_title"], "page_count": len(pages), "processed": processed},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if processed % args.checkpoint_every == 0:
            summarize(args.output_dir, output_jsonl)
    summarize(args.output_dir, output_jsonl)


if __name__ == "__main__":
    main()
