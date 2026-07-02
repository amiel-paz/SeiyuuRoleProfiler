#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watch_safe_enrichment_llm_tags import (
    TAG_CATEGORIES,
    append_jsonl,
    normalize_tag_payload,
    parse_json_object,
    prompt_for_row,
    read_complete_input_rows,
    row_allowed,
    source_blocks,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-tag safe-enrichment rows through a vLLM chat-completions-compatible endpoint."
    )
    parser.add_argument("--input", type=Path, default=Path("data/external/safe_enrichment/character_safe_enrichment.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/external/safe_enrichment_llm/character_tags.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("data/external/safe_enrichment_llm/errors.jsonl"))
    parser.add_argument("--raw-cache-dir", type=Path, default=Path("data/external/safe_enrichment_llm/raw_vllm"))
    parser.add_argument("--status", type=Path, default=Path("data/external/safe_enrichment_llm/status_vllm.json"))
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--structured-mode", choices=("none", "json_object", "json_schema"), default="json_object")
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--enable-thinking", dest="disable_thinking", action="store_false")
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


def row_id(row: dict) -> int:
    return int(row["anilist_character_id"])


def raw_cache_path(args: argparse.Namespace, row: dict) -> Path:
    return args.raw_cache_dir / f"{row_id(row)}.json"


def processed_ids(path: Path) -> set[int]:
    ids = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                ids.add(int(payload["anilist_character_id"]))
            except Exception:
                continue
    return ids


def vllm_schema() -> dict:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tag": {"type": "string"},
            "evidence": {"type": "string"},
            "source_key": {"type": "string"},
            "source_url": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["tag", "evidence", "source_key", "source_url", "confidence"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {category: {"type": "array", "items": item} for category in TAG_CATEGORIES},
        "required": list(TAG_CATEGORIES),
    }


def vllm_payload(args: argparse.Namespace, prompt: str) -> dict:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a strict information-extraction assistant. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "top_k": args.top_k,
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if args.structured_mode == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif args.structured_mode == "json_schema":
        payload["structured_outputs"] = {"json": vllm_schema()}
    return payload


def call_vllm(args: argparse.Namespace, prompt: str) -> dict:
    request = urllib.request.Request(
        f"{args.vllm_url.rstrip('/')}/chat/completions",
        data=json.dumps(vllm_payload(args, prompt)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def response_content(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_block_summary(blocks: list[dict]) -> list[dict]:
    return [{key: block[key] for key in ("source_key", "source", "url", "license")} for block in blocks]


def tag_row(row: dict, blocks: list[dict], args: argparse.Namespace) -> dict:
    path = raw_cache_path(args, row)
    prompt = prompt_for_row(row, blocks, args.max_tags_per_category)
    if path.exists() and not args.refresh_raw:
        cached = read_json(path)
    else:
        response = call_vllm(args, prompt)
        cached = {
            "generated_at": utc_now(),
            "vllm_model": args.model,
            "options": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "presence_penalty": args.presence_penalty,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "structured_mode": args.structured_mode,
                "disable_thinking": args.disable_thinking,
            },
            "local_character": {
                "anilist_character_id": row["anilist_character_id"],
                "name": row["name"],
                "native_name": row.get("native_name") or "",
                "first_anime": row.get("first_anime") or "",
                "favourites": row.get("favourites"),
                "site_url": row.get("site_url") or "",
            },
            "source_blocks": source_block_summary(blocks),
            "prompt": prompt,
            "response": response,
        }
        write_json(path, cached)
    parsed = normalize_tag_payload(parse_json_object(response_content(cached.get("response") or {})), blocks)
    return {
        "generated_at": utc_now(),
        "tagger_runtime": "vllm",
        "tagger_model": args.model,
        "anilist_character_id": row["anilist_character_id"],
        "name": row["name"],
        "native_name": row.get("native_name") or "",
        "first_anime": row.get("first_anime") or "",
        "favourites": row.get("favourites"),
        "source_block_count": len(blocks),
        "source_blocks": source_block_summary(blocks),
        "tags": parsed,
    }


def candidate_rows(args: argparse.Namespace) -> tuple[list[tuple[dict, list[dict]]], int, int]:
    done = processed_ids(args.output) if not args.force else set()
    failed = processed_ids(args.errors) if not args.force else set()
    skipped_no_text = 0
    rows = []
    for row in read_complete_input_rows(args.input):
        cid = row_id(row)
        if cid in done or cid in failed:
            continue
        if not row_allowed(row, args):
            continue
        blocks = source_blocks(row, args.max_source_chars)
        total_chars = sum(len(block["text"]) for block in blocks)
        if total_chars < args.min_source_chars:
            skipped_no_text += 1
            continue
        rows.append((row, blocks))
        if args.max_rows_per_pass and len(rows) >= args.max_rows_per_pass:
            break
    return rows, len(done), skipped_no_text


def process_one(row: dict, blocks: list[dict], args: argparse.Namespace) -> tuple[str, dict]:
    try:
        tagged = tag_row(row, blocks, args)
    except (json.JSONDecodeError, ValueError) as error:
        return (
            "error",
            {
                "generated_at": utc_now(),
                "anilist_character_id": row_id(row),
                "name": row.get("name"),
                "error": repr(error),
                "raw_cache": str(raw_cache_path(args, row)),
                "source_block_count": len(blocks),
            },
        )
    return "success", tagged


def run_pass(args: argparse.Namespace) -> dict:
    rows, done_before, skipped_no_text = candidate_rows(args)
    processed = 0
    failed = 0
    started_at = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(process_one, row, blocks, args) for row, blocks in rows]
        for future in concurrent.futures.as_completed(futures):
            kind, payload = future.result()
            if kind == "success":
                append_jsonl(args.output, payload)
                processed += 1
                print(
                    json.dumps(
                        {
                            "id": payload["anilist_character_id"],
                            "name": payload.get("name"),
                            "source_blocks": payload["source_block_count"],
                            "role": len(payload["tags"]["role"]),
                            "personality": len(payload["tags"]["personality"]),
                            "traits": len(payload["tags"]["traits"]),
                            "processed_this_pass": processed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                append_jsonl(args.errors, payload)
                failed += 1
                print(
                    json.dumps(
                        {
                            "id": payload["anilist_character_id"],
                            "name": payload.get("name"),
                            "error": "parse_failed",
                            "details": payload["error"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            status = {
                "generated_at": utc_now(),
                "runtime": "vllm",
                "model": args.model,
                "input_rows_seen": len(rows) + done_before,
                "output_rows_done_estimate": done_before + processed,
                "error_rows_this_pass": failed,
                "processed_this_pass": processed,
                "eligible_unprocessed_seen": len(rows),
                "skipped_no_text_this_pass": skipped_no_text,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
            write_json(args.status, status)
    status = {
        "generated_at": utc_now(),
        "runtime": "vllm",
        "model": args.model,
        "input_rows_seen": len(rows) + done_before,
        "output_rows_done_estimate": done_before + processed,
        "error_rows_this_pass": failed,
        "processed_this_pass": processed,
        "eligible_unprocessed_seen": len(rows),
        "skipped_no_text_this_pass": skipped_no_text,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
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
                        "output_rows_done": status["output_rows_done_estimate"],
                        "sleep": args.sleep_when_caught_up,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(args.sleep_when_caught_up)


if __name__ == "__main__":
    main()
