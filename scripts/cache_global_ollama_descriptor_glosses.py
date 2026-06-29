#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_translation_regularized_descriptor_distance import encode_bge_small, short_phrase_variants
from scripts.seiyuu_local_orthogonal_svd import character_descriptors, slug
from scripts.seiyuu_ollama_gloss_svd import extract_json, ollama_generate, prompt_for_descriptor, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Ollama semantic glosses for all unique filtered LLM descriptors.")
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_only_personality_traits.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/global_ollama_descriptor_glosses"))
    parser.add_argument("--categories", nargs="+", default=["personality", "traits"])
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-predict", type=int, default=260)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--max-variant-words", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def collect_descriptors(tags_input: Path, categories: set[str]) -> list[str]:
    payload = read_json(tags_input)
    descriptors: set[str] = set()
    for character in payload.get("characters", []):
        descriptors.update(character_descriptors(character, categories))
    return sorted(descriptors)


def cache_base(args: argparse.Namespace, descriptor_count: int) -> Path:
    categories = "_".join(sorted(args.categories))
    suffix = f"n{descriptor_count}" if args.limit else "all"
    return args.output_dir / (
        f"{slug(args.tags_input.stem)}_{slug(args.ollama_model)}_{slug(categories)}_filtered_{suffix}"
    )


def completed_rows(path: Path, descriptors: list[str], force: bool) -> dict[str, dict]:
    if force or not path.exists():
        return {}
    payload = read_json(path)
    wanted = set(descriptors)
    return {
        str(row.get("descriptor")): row
        for row in payload.get("rows", [])
        if str(row.get("descriptor") or "") in wanted
    }


def write_checkpoint(path: Path, args: argparse.Namespace, descriptors: list[str], rows_by_descriptor: dict[str, dict]) -> None:
    rows = [rows_by_descriptor[descriptor] for descriptor in descriptors if descriptor in rows_by_descriptor]
    write_json(
        path,
        {
            "generated_at": utc_now(),
            "complete": len(rows) == len(descriptors),
            "source": "cache_global_ollama_descriptor_glosses.py",
            "parameters": {
                "tags_input": str(args.tags_input),
                "categories": sorted(args.categories),
                "ollama_url": args.ollama_url,
                "ollama_model": args.ollama_model,
                "temperature": args.temperature,
                "seed": args.seed,
                "num_predict": args.num_predict,
                "num_ctx": args.num_ctx,
                "max_variant_words": args.max_variant_words,
                "descriptor_filtering": (
                    "Uses seiyuu_local_orthogonal_svd.character_descriptors, including metadata-field "
                    "and finite-verb-head exclusions."
                ),
            },
            "counts": {
                "descriptors": len(descriptors),
                "glossed": len(rows),
                "remaining": len(descriptors) - len(rows),
            },
            "descriptors": descriptors,
            "rows": rows,
        },
    )


def build_embeddings(json_path: Path, npz_path: Path) -> None:
    payload = read_json(json_path)
    if not payload.get("complete"):
        raise RuntimeError(f"Cannot embed incomplete gloss cache: {json_path}")
    variant_texts = []
    for row in payload["rows"]:
        glosses = list(row["glosses"])
        while len(glosses) < 4:
            glosses.append(glosses[-1] if glosses else row["descriptor"])
        variant_texts.extend(glosses[:4])
    variant_embeddings = encode_bge_small(variant_texts).reshape(len(payload["rows"]), 4, -1).astype(np.float64)
    np.savez_compressed(npz_path, variant_embeddings=variant_embeddings)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    categories = {category.strip() for category in args.categories}
    descriptors = collect_descriptors(args.tags_input, categories)
    if args.limit:
        descriptors = descriptors[: args.limit]
    base = cache_base(args, len(descriptors))
    json_path = base.with_name(f"{base.name}_ollama_glosses.json")
    npz_path = base.with_name(f"{base.name}_ollama_glosses.npz")
    rows_by_descriptor = completed_rows(json_path, descriptors, args.force)

    ollama_args = SimpleNamespace(
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        temperature=args.temperature,
        seed=args.seed,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
    )
    print(f"descriptor_count={len(descriptors)} already_glossed={len(rows_by_descriptor)}", flush=True)
    for index, descriptor in enumerate(descriptors, 1):
        if descriptor in rows_by_descriptor:
            continue
        raw_payload, text = ollama_generate(ollama_args, prompt_for_descriptor(descriptor))
        try:
            parsed = extract_json(text)
            raw_glosses = [str(value) for value in parsed.get("glosses", [])]
        except Exception:
            raw_glosses = []
        glosses = short_phrase_variants(raw_glosses, descriptor, args.max_variant_words)
        padded = glosses[:4]
        while len(padded) < 4:
            padded.append(padded[-1] if padded else descriptor)
        rows_by_descriptor[descriptor] = {
            "descriptor": descriptor,
            "glosses": padded,
            "raw_text": text,
            "done_reason": raw_payload.get("done_reason"),
            "eval_count": raw_payload.get("eval_count"),
        }
        if len(rows_by_descriptor) % args.checkpoint_every == 0 or index == len(descriptors):
            write_checkpoint(json_path, args, descriptors, rows_by_descriptor)
            print(f"glossed {len(rows_by_descriptor)}/{len(descriptors)}", flush=True)

    write_checkpoint(json_path, args, descriptors, rows_by_descriptor)
    print(f"wrote {json_path}", flush=True)
    if not args.skip_embeddings:
        build_embeddings(json_path, npz_path)
        print(f"wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
