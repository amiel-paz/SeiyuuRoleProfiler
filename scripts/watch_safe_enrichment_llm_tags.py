#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TAG_CATEGORIES = ("role", "personality", "traits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously LLM-tag completed safe-enrichment rows as they are produced."
    )
    parser.add_argument("--input", type=Path, default=Path("data/external/safe_enrichment/character_safe_enrichment.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/external/safe_enrichment_llm/character_tags.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("data/external/safe_enrichment_llm/errors.jsonl"))
    parser.add_argument("--raw-cache-dir", type=Path, default=Path("data/external/safe_enrichment_llm/raw"))
    parser.add_argument("--status", type=Path, default=Path("data/external/safe_enrichment_llm/status.json"))
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--think", choices=("false", "true", "low", "medium", "high", "max"), default="false")
    parser.add_argument("--sleep-when-caught-up", type=float, default=60.0)
    parser.add_argument("--sleep-after-error", type=float, default=30.0)
    parser.add_argument("--min-source-chars", type=int, default=80)
    parser.add_argument("--max-source-chars", type=int, default=5000)
    parser.add_argument("--max-tags-per-category", type=int, default=8)
    parser.add_argument("--max-rows-per-pass", type=int, default=0, help="0 means no pass limit.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--name", action="append", default=[])
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def error_ids(path: Path) -> set[int]:
    return processed_ids(path)


def read_complete_input_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                continue
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def truncate(value: str, max_chars: int) -> str:
    value = clean_text(value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def source_blocks(row: dict, max_chars: int) -> list[dict]:
    blocks = []
    for match in (row.get("bangumi") or {}).get("matches") or []:
        text = truncate(match.get("summary") or "", max_chars)
        if text:
            blocks.append(
                {
                    "source_key": f"bangumi:{match.get('bangumi_character_id')}",
                    "source": "bangumi",
                    "url": match.get("url") or "",
                    "license": (match.get("policy") or {}).get("license") or "",
                    "text": text,
                }
            )
    for match in (row.get("wikidata") or {}).get("matches") or []:
        trait_labels = [trait.get("label") for trait in match.get("traits") or [] if trait.get("label")]
        text_parts = [match.get("description") or ""]
        if trait_labels:
            text_parts.append("Wikidata traits: " + ", ".join(trait_labels))
        text = truncate(" ".join(part for part in text_parts if part), max_chars)
        if text:
            blocks.append(
                {
                    "source_key": f"wikidata:{match.get('qid')}",
                    "source": "wikidata",
                    "url": match.get("url") or "",
                    "license": (match.get("policy") or {}).get("license") or "CC0",
                    "text": text,
                }
            )
    for key in ("wikipedia", "wikipedia_broad"):
        for match in (row.get(key) or {}).get("matches") or []:
            text = match.get("extract") or ""
            snippets = match.get("mention_snippets") or []
            if snippets:
                text = " ".join(snippet.get("snippet") or "" for snippet in snippets if snippet.get("snippet")) or text
            text = truncate(text, max_chars)
            if text:
                blocks.append(
                    {
                        "source_key": f"{match.get('source')}:{match.get('title')}",
                        "source": match.get("source") or key,
                        "url": match.get("url") or "",
                        "license": (match.get("policy") or {}).get("license") or "",
                        "text": text,
                    }
                )
    return blocks


def prompt_for_row(row: dict, blocks: list[dict], max_tags_per_category: int) -> str:
    source_text = "\n\n".join(
        (
            f"[{index}] source_key: {block['source_key']}\n"
            f"source: {block['source']}\n"
            f"url: {block['url']}\n"
            f"license: {block['license']}\n"
            f"text: {block['text']}"
        )
        for index, block in enumerate(blocks, start=1)
    )
    return f"""Extract structured character tags from source-attributed anime character descriptions.

Use only the supplied source blocks. Do not use outside knowledge, image cues, or the character name alone.
Every tag must be directly supported by an exact evidence span copied from one source block.

Categories:
- role: social, story, family, school, job, team, relationship, or narrative roles.
- personality: stable temperament, attitude, behavior style, or interpersonal style.
- traits: other stable non-appearance traits, abilities, interests, habits, skills, or conditions.

Rules:
- Each category is an array and may contain zero, one, or many tags.
- Return at most {max_tags_per_category} tags per category. Keep the strongest, clearest tags only.
- The tag field must be a canonical English phrase of 1 to 4 words.
- Prefer adjective/noun phrases. Avoid verbs, pronouns, conjunctions, prepositions, determiners, and adverbs in tags.
- Split combined concepts into separate entries.
- Convert supported clauses into adjective/noun tags when possible.
- Do not include hair, eye, outfit, body-shape, measurements, or other appearance tags.
- Do not include plot-only events unless they imply a stable role or trait.
- Do not include duplicate or near-duplicate tags.
- If a category has no directly supported tags, return an empty array.
- If uncertain, omit.

Character metadata:
- id: {row['anilist_character_id']}
- name: {row['name']}
- native name: {row.get('native_name') or ''}
- first anime: {row.get('first_anime') or ''}

Source blocks:
{source_text}

Return only JSON with this shape:
{{
  "role": [
    {{"tag": "short tag", "evidence": "exact text span", "source_key": "source key", "source_url": "url", "confidence": "high|medium|low"}}
  ],
  "personality": [
    {{"tag": "short tag", "evidence": "exact text span", "source_key": "source key", "source_url": "url", "confidence": "high|medium|low"}}
  ],
  "traits": [
    {{"tag": "short tag", "evidence": "exact text span", "source_key": "source key", "source_url": "url", "confidence": "high|medium|low"}}
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


def call_ollama(args: argparse.Namespace, prompt: str) -> dict:
    think: bool | str = args.think
    if args.think == "false":
        think = False
    elif args.think == "true":
        think = True
    payload = {
        "model": args.ollama_model,
        "messages": [
            {"role": "system", "content": "You are a strict information-extraction assistant. Return valid JSON only."},
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
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def raw_cache_path(args: argparse.Namespace, row: dict) -> Path:
    return args.raw_cache_dir / f"{row['anilist_character_id']}.json"


def normalize_source_key(source_key: str, blocks: list[dict]) -> str:
    source_key = clean_text(source_key)
    match = re.fullmatch(r"\[(\d+)\]", source_key)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(blocks):
            return blocks[index]["source_key"]
    return source_key


def normalize_source_url(source_url: str, source_key: str, blocks: list[dict]) -> str:
    source_url = clean_text(source_url)
    if source_url:
        return source_url
    for block in blocks:
        if block["source_key"] == source_key:
            return block["url"]
    return ""


def normalize_tag_payload(payload: dict, blocks: list[dict]) -> dict:
    normalized = {}
    for category in TAG_CATEGORIES:
        rows = payload.get(category) or []
        if not isinstance(rows, list):
            rows = []
        cleaned = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            tag = clean_text(str(item.get("tag") or "")).lower()
            evidence = clean_text(str(item.get("evidence") or ""))
            if not tag or not evidence:
                continue
            source_key = normalize_source_key(str(item.get("source_key") or ""), blocks)
            cleaned.append(
                {
                    "tag": tag,
                    "evidence": evidence,
                    "source_key": source_key,
                    "source_url": normalize_source_url(str(item.get("source_url") or ""), source_key, blocks),
                    "confidence": item.get("confidence") if item.get("confidence") in {"high", "medium", "low"} else "low",
                }
            )
        normalized[category] = cleaned
    return normalized


def tag_row(row: dict, blocks: list[dict], args: argparse.Namespace) -> dict:
    path = raw_cache_path(args, row)
    prompt = prompt_for_row(row, blocks, args.max_tags_per_category)
    if path.exists() and not args.refresh_raw:
        cached = read_json(path)
    else:
        response = call_ollama(args, prompt)
        cached = {
            "generated_at": utc_now(),
            "ollama_model": args.ollama_model,
            "options": {
                "temperature": args.temperature,
                "seed": args.seed,
                "num_predict": args.num_predict,
                "num_ctx": args.num_ctx,
                "think": args.think,
            },
            "local_character": {
                "anilist_character_id": row["anilist_character_id"],
                "name": row["name"],
                "native_name": row.get("native_name") or "",
                "first_anime": row.get("first_anime") or "",
                "favourites": row.get("favourites"),
                "site_url": row.get("site_url") or "",
            },
            "source_blocks": [{key: block[key] for key in ("source_key", "source", "url", "license")} for block in blocks],
            "prompt": prompt,
            "response": response,
        }
        write_json(path, cached)
    content = ((cached.get("response") or {}).get("message") or {}).get("content") or ""
    parsed = normalize_tag_payload(parse_json_object(content), blocks)
    return {
        "generated_at": utc_now(),
        "anilist_character_id": row["anilist_character_id"],
        "name": row["name"],
        "native_name": row.get("native_name") or "",
        "first_anime": row.get("first_anime") or "",
        "favourites": row.get("favourites"),
        "source_block_count": len(blocks),
        "source_blocks": [{key: block[key] for key in ("source_key", "source", "url", "license")} for block in blocks],
        "tags": parsed,
    }


def row_allowed(row: dict, args: argparse.Namespace) -> bool:
    if not args.name:
        return True
    requested = set().union(*(name_keys(value) for value in args.name))
    return bool(name_keys(row.get("name") or "").intersection(requested))


def run_pass(args: argparse.Namespace) -> dict:
    done = processed_ids(args.output) if not args.force else set()
    failed = error_ids(args.errors) if not args.force else set()
    rows = read_complete_input_rows(args.input)
    processed = 0
    skipped_no_text = 0
    candidates = 0
    for row in rows:
        cid = int(row["anilist_character_id"])
        if cid in done or cid in failed:
            continue
        if not row_allowed(row, args):
            continue
        blocks = source_blocks(row, args.max_source_chars)
        total_chars = sum(len(block["text"]) for block in blocks)
        if total_chars < args.min_source_chars:
            skipped_no_text += 1
            continue
        candidates += 1
        try:
            tagged = tag_row(row, blocks, args)
        except (json.JSONDecodeError, ValueError) as error:
            append_jsonl(
                args.errors,
                {
                    "generated_at": utc_now(),
                    "anilist_character_id": cid,
                    "name": row.get("name"),
                    "error": repr(error),
                    "raw_cache": str(raw_cache_path(args, row)),
                    "source_block_count": len(blocks),
                },
            )
            failed.add(cid)
            print(
                json.dumps(
                    {"id": cid, "name": row.get("name"), "error": "parse_failed", "details": repr(error)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        append_jsonl(args.output, tagged)
        done.add(cid)
        processed += 1
        print(
            json.dumps(
                {
                    "id": cid,
                    "name": row.get("name"),
                    "source_blocks": len(blocks),
                    "role": len(tagged["tags"]["role"]),
                    "personality": len(tagged["tags"]["personality"]),
                    "traits": len(tagged["tags"]["traits"]),
                    "processed_this_pass": processed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.max_rows_per_pass and processed >= args.max_rows_per_pass:
            break
    status = {
        "generated_at": utc_now(),
        "input_rows_seen": len(rows),
        "output_rows_done": len(done),
        "error_rows_done": len(failed),
        "processed_this_pass": processed,
        "eligible_unprocessed_seen": candidates,
        "skipped_no_text_this_pass": skipped_no_text,
    }
    write_json(args.status, status)
    return status


def main() -> None:
    args = parse_args()
    while True:
        try:
            status = run_pass(args)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            print(json.dumps({"error": repr(error), "sleep": args.sleep_after_error}, ensure_ascii=False), flush=True)
            time.sleep(args.sleep_after_error)
            if args.once:
                raise
            continue
        if args.once:
            break
        if status["processed_this_pass"] == 0:
            print(
                json.dumps(
                    {
                        "caught_up": True,
                        "input_rows_seen": status["input_rows_seen"],
                        "output_rows_done": status["output_rows_done"],
                        "sleep": args.sleep_when_caught_up,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(args.sleep_when_caught_up)


if __name__ == "__main__":
    main()
