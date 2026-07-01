#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watch_safe_enrichment_llm_tags import (
    normalize_tag_payload,
    parse_json_object,
    prompt_for_row,
    read_complete_input_rows,
    source_blocks,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched CUDA Transformers runner for safe-enrichment LLM tags.")
    parser.add_argument("--input", type=Path, default=Path("data/pending_safe_enrichment.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/batch_transformers/character_tags.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("output/batch_transformers/errors.jsonl"))
    parser.add_argument("--raw-cache-dir", type=Path, default=Path("output/batch_transformers/raw"))
    parser.add_argument("--status", type=Path, default=Path("output/batch_transformers/status.json"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_CACHE_DIR", "/dev/shm/seiyuu_transformers_cache/hf"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-source-chars", type=int, default=5000)
    parser.add_argument("--max-tags-per-category", type=int, default=8)
    parser.add_argument("--min-source-chars", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--sort-by-prompt-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compact-prompt", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def processed_ids(path: Path) -> set[int]:
    ids: set[int] = set()
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


def source_block_summary(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: block.get(key, "") for key in ("source_key", "source", "url", "license")} for block in blocks]


def raw_cache_path(raw_cache_dir: Path, row: dict[str, Any]) -> Path:
    return raw_cache_dir / f"{int(row['anilist_character_id'])}.json"


def repair_unescaped_inner_quotes(value: str) -> str:
    output: list[str] = []
    in_string = False
    escape = False
    length = len(value)
    for index, char in enumerate(value):
        if escape:
            output.append(char)
            escape = False
            continue
        if char == "\\":
            output.append(char)
            escape = True
            continue
        if char == '"' and in_string:
            next_index = index + 1
            while next_index < length and value[next_index].isspace():
                next_index += 1
            if next_index >= length or value[next_index] in ",}]:":
                in_string = False
                output.append(char)
            else:
                output.append('\\"')
            continue
        if char == '"':
            in_string = True
            output.append(char)
            continue
        output.append(char)
    return "".join(output)


def parse_model_json(value: str) -> dict[str, Any]:
    try:
        return parse_json_object(value)
    except Exception:
        return parse_json_object(repair_unescaped_inner_quotes(value))


def compact_prompt_for_row(row: dict[str, Any], blocks: list[dict[str, Any]], max_tags_per_category: int) -> str:
    source_text = "\n".join(
        f"[{index}] key={block['source_key']} source={block['source']} url={block['url']} text={block['text']}"
        for index, block in enumerate(blocks, start=1)
    )
    return f"""Extract character tags from only these source blocks.
Return minified valid JSON only. No markdown. No comments.
Categories:
- role: social, story, family, school, job, team, relationship, or narrative roles.
- personality: stable temperament, attitude, behavior style, interpersonal style.
- traits: stable non-appearance traits, abilities, interests, habits, skills, conditions.
Rules:
- At most {max_tags_per_category} strongest tags per category.
- tag: canonical English adjective/noun phrase, 1-4 words.
- evidence: shortest exact source span, preferably under 80 characters.
- evidence must not contain double-quote characters; choose a shorter span without them.
- source_key: use the block key exactly.
- confidence: high, medium, or low.
- Avoid verbs, pronouns, conjunctions, prepositions, determiners, adverbs.
- Exclude hair, eyes, outfit, body shape, measurements, and other appearance.
- If uncertain, omit.
Character: id={row['anilist_character_id']} name={row['name']} native={row.get('native_name') or ''} anime={row.get('first_anime') or ''}
Sources:
{source_text}
JSON shape: {{"role":[{{"tag":"","evidence":"","source_key":"","source_url":"","confidence":""}}],"personality":[{{"tag":"","evidence":"","source_key":"","source_url":"","confidence":""}}],"traits":[{{"tag":"","evidence":"","source_key":"","source_url":"","confidence":""}}]}}"""


def prepare_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    done = set() if args.force else processed_ids(args.output)
    failed = set() if args.force else processed_ids(args.errors)
    rows: list[dict[str, Any]] = []
    for row in read_complete_input_rows(args.input):
        cid = int(row["anilist_character_id"])
        if cid in done or cid in failed:
            continue
        blocks = source_blocks(row, args.max_source_chars)
        if sum(len(block["text"]) for block in blocks) < args.min_source_chars:
            continue
        if args.compact_prompt:
            prompt = compact_prompt_for_row(row, blocks, args.max_tags_per_category)
        else:
            prompt = prompt_for_row(row, blocks, args.max_tags_per_category)
        rows.append({"row": row, "blocks": blocks, "prompt": prompt, "prompt_chars": len(prompt)})
        if args.limit and len(rows) >= args.limit:
            break
    if args.sort_by_prompt_length:
        rows.sort(key=lambda item: item["prompt_chars"])
    return rows


def make_chat_text(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a strict information-extraction assistant. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def write_status(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    write_json(args.status, {"generated_at": utc_now(), **payload})


def main() -> None:
    args = parse_args()
    args.raw_cache_dir.mkdir(parents=True, exist_ok=True)
    rows = prepare_rows(args)
    write_status(
        args,
        {
            "event": "starting",
            "pending_rows": len(rows),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "model": args.model,
        },
    )
    print(
        json.dumps(
            {
                "event": "starting",
                "pending_rows": len(rows),
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "model": args.model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(json.dumps({"event": "loading", "torch": torch.__version__, "cuda": torch.cuda.is_available()}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as error:
            print(json.dumps({"event": "compile_failed", "error": repr(error)}), flush=True)
    print(json.dumps({"event": "loaded"}), flush=True)

    processed = 0
    errors = 0
    t_start = time.time()
    token_count = 0
    generated_token_count = 0

    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        texts = [make_chat_text(tokenizer, item["prompt"]) for item in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to("cuda")
        prompt_tokens = int(inputs.input_ids.numel())
        t0 = time.time()
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            batch_seconds = time.time() - t0
            new_tokens = int(max(0, generated.shape[1] - inputs.input_ids.shape[1]) * len(batch))
            token_count += prompt_tokens
            generated_token_count += new_tokens
            decoded = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        except Exception as error:
            batch_seconds = time.time() - t0
            for item in batch:
                row = item["row"]
                cid = int(row["anilist_character_id"])
                append_jsonl(
                    args.errors,
                    {
                        "generated_at": utc_now(),
                        "anilist_character_id": cid,
                        "name": row.get("name"),
                        "error": repr(error),
                        "stage": "batch_generate",
                    },
                )
                errors += 1
            print(
                json.dumps(
                    {
                        "event": "batch_error",
                        "batch_start": batch_start,
                        "batch_size": len(batch),
                        "seconds": round(batch_seconds, 3),
                        "error": repr(error),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        for item, response_text in zip(batch, decoded, strict=True):
            row = item["row"]
            blocks = item["blocks"]
            cid = int(row["anilist_character_id"])
            raw_path = raw_cache_path(args.raw_cache_dir, row)
            raw_payload = {
                "generated_at": utc_now(),
                "runtime": "transformers_batch_cuda",
                "model": args.model,
                "anilist_character_id": cid,
                "name": row.get("name"),
                "response_text": response_text,
            }
            if args.refresh_raw or not raw_path.exists():
                write_json(raw_path, raw_payload)
            try:
                parsed = normalize_tag_payload(parse_model_json(response_text), blocks)
            except Exception as error:
                append_jsonl(
                    args.errors,
                    {
                        "generated_at": utc_now(),
                        "anilist_character_id": cid,
                        "name": row.get("name"),
                        "error": repr(error),
                        "raw_cache": str(raw_path),
                        "response_preview": response_text[:500],
                    },
                )
                errors += 1
                continue
            append_jsonl(
                args.output,
                {
                    "generated_at": utc_now(),
                    "tagger_runtime": "transformers_batch_cuda",
                    "tagger_model": args.model,
                    "anilist_character_id": cid,
                    "name": row.get("name"),
                    "native_name": row.get("native_name") or "",
                    "first_anime": row.get("first_anime") or "",
                    "favourites": row.get("favourites"),
                    "source_block_count": len(blocks),
                    "source_blocks": source_block_summary(blocks),
                    "tags": parsed,
                },
            )
            processed += 1

        elapsed = time.time() - t_start
        status = {
            "event": "running",
            "processed": processed,
            "errors": errors,
            "remaining": max(0, len(rows) - batch_start - len(batch)),
            "elapsed_seconds": round(elapsed, 2),
            "rows_per_minute": round(processed / max(elapsed, 1e-9) * 60.0, 2),
            "last_batch_seconds": round(batch_seconds, 3),
            "last_batch_size": len(batch),
            "prompt_tokens_total_estimate": token_count,
            "generated_tokens_total_estimate": generated_token_count,
        }
        write_status(args, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    elapsed = time.time() - t_start
    final_status = {
        "event": "done",
        "processed": processed,
        "errors": errors,
        "remaining": 0,
        "elapsed_seconds": round(elapsed, 2),
        "rows_per_minute": round(processed / max(elapsed, 1e-9) * 60.0, 2),
        "prompt_tokens_total_estimate": token_count,
        "generated_tokens_total_estimate": generated_token_count,
    }
    write_status(args, final_status)
    print(json.dumps(final_status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
