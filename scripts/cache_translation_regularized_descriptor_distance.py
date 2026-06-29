#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_METADATA = Path(
    "models/tag_descriptor_matrices_llm_only/all_characters_llm_only_personality_traits_matrix_metadata.json"
)
DEFAULT_BGE_SMALL = Path("models/embedding_bakeoff/bge_small_descriptor_embeddings.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache translation-regularized descriptor similarity scores.")
    parser.add_argument("--matrix-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--bge-small-embeddings", type=Path, default=DEFAULT_BGE_SMALL)
    parser.add_argument("--output-dir", type=Path, default=Path("models/translation_regularized_descriptors"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--descriptors", nargs="*", default=None)
    parser.add_argument("--translation-backend", choices=["marian", "nllb"], default="nllb")
    parser.add_argument("--en-ja-model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--ja-en-model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--en-lang", default="eng_Latn")
    parser.add_argument("--ja-lang", default="jpn_Jpan")
    parser.add_argument("--ja-beams", type=int, default=8)
    parser.add_argument("--ja-top", type=int, default=2)
    parser.add_argument("--en-beams", type=int, default=8)
    parser.add_argument("--en-top-per-ja", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-variant-words", type=int, default=4)
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


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        normalized = normalize_space(value)
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def short_phrase_variants(values: list[str], fallback: str, max_words: int) -> list[str]:
    output = []
    for value in values:
        normalized = normalize_space(value)
        normalized = normalized.strip(" .。!?！？\"'“”‘’")
        word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", normalized))
        if 0 < word_count <= max_words:
            output.append(normalized)
    output = unique_keep_order(output)
    return output or [fallback]


def load_translation(model_name: str, backend: str, source_lang: str | None, target_lang: str | None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_lang) if backend == "nllb" else AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return {"backend": backend, "tokenizer": tokenizer, "model": model, "source_lang": source_lang, "target_lang": target_lang}


def translate_batch(
    texts: list[str],
    translator: dict[str, Any],
    num_beams: int,
    num_return_sequences: int,
    max_new_tokens: int,
) -> list[list[str]]:
    tokenizer = translator["tokenizer"]
    model = translator["model"]
    if translator["backend"] == "nllb":
        tokenizer.src_lang = translator["source_lang"]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    generate_kwargs = {}
    if translator["backend"] == "nllb":
        generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(translator["target_lang"])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    rows = []
    for offset in range(0, len(decoded), num_return_sequences):
        rows.append(unique_keep_order(decoded[offset : offset + num_return_sequences]))
    return rows


def load_bge_small() -> tuple[list[str], np.ndarray]:
    metadata = read_json(Path("models/embedding_bakeoff/bge_small_descriptor_embeddings.json"))
    embeddings = np.load(DEFAULT_BGE_SMALL)["embeddings"].astype(np.float32)
    return metadata["descriptors"], embeddings


def encode_bge_small(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def coherence_from_embeddings(variants: list[str], embeddings: np.ndarray) -> dict[str, Any]:
    target_count = 4
    similarity = embeddings @ embeddings.T
    eigenvalues = np.linalg.eigvalsh(similarity)
    lambda_max = float(eigenvalues[-1])
    coherence = lambda_max / target_count
    return {
        "english_variants": variants,
        "similarity_matrix": np.round(similarity, 8).tolist(),
        "eigenvalues": np.round(eigenvalues, 8).tolist(),
        "lambda_max": round(lambda_max, 8),
        "coherence": round(coherence, 8),
        "distance": round(1.0 - coherence, 8),
    }


def regularized_pair_similarity(left: np.ndarray, right: np.ndarray) -> float:
    cross_similarity = left @ right.T
    return float(np.linalg.svd(cross_similarity, compute_uv=False)[0] / left.shape[0])


def main() -> None:
    args = parse_args()
    metadata = read_json(args.matrix_metadata)
    all_descriptors = metadata["descriptors"]
    if args.descriptors:
        wanted = {descriptor.casefold() for descriptor in args.descriptors}
        descriptors = [descriptor for descriptor in all_descriptors if descriptor.casefold() in wanted]
        missing = sorted(wanted - {descriptor.casefold() for descriptor in descriptors})
        if missing:
            raise RuntimeError(f"Descriptors not found: {missing}")
    else:
        descriptors = all_descriptors[: args.limit] if args.limit else all_descriptors

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all" if not args.descriptors and not args.limit else f"n{len(descriptors)}"
    translator_slug = slug(f"{args.translation_backend}_{args.en_ja_model}_{args.ja_en_model}")
    output_path = args.output_dir / f"translation_regularized_bge_small_{translator_slug}_{suffix}.json"
    if output_path.exists() and not args.force:
        print(f"exists: {output_path}")
        return

    en_ja = load_translation(args.en_ja_model, args.translation_backend, args.en_lang, args.ja_lang)
    ja_en = load_translation(args.ja_en_model, args.translation_backend, args.ja_lang, args.en_lang)

    ja_by_descriptor = []
    for start in range(0, len(descriptors), args.batch_size):
        batch = descriptors[start : start + args.batch_size]
        ja_by_descriptor.extend(translate_batch(batch, en_ja, args.ja_beams, args.ja_top, args.max_new_tokens))
        print(f"translated en->ja {min(start + args.batch_size, len(descriptors))}/{len(descriptors)}", flush=True)

    flat_ja = [term for row in ja_by_descriptor for term in row[: args.ja_top]]
    flat_back_rows = []
    for start in range(0, len(flat_ja), args.batch_size):
        batch = flat_ja[start : start + args.batch_size]
        flat_back_rows.extend(translate_batch(batch, ja_en, args.en_beams, args.en_top_per_ja, args.max_new_tokens))
        print(f"translated ja->en {min(start + args.batch_size, len(flat_ja))}/{len(flat_ja)}", flush=True)

    rows = []
    variant_texts = []
    cursor = 0
    for descriptor, ja_translations in zip(descriptors, ja_by_descriptor):
        selected_ja = ja_translations[: args.ja_top]
        back_rows = flat_back_rows[cursor : cursor + len(selected_ja)]
        cursor += len(selected_ja)
        english_variants = short_phrase_variants(
            [value for row in back_rows for value in row],
            descriptor,
            args.max_variant_words,
        )
        padded = english_variants[:4]
        while len(padded) < 4:
            padded.append(padded[-1] if padded else "")
        rows.append(
            {
                "descriptor": descriptor,
                "japanese_translations": selected_ja,
                "english_variants": padded,
            }
        )
        variant_texts.extend(padded)

    variant_embeddings = encode_bge_small(variant_texts).reshape(len(rows), 4, -1)
    for index, row in enumerate(rows):
        row.update(coherence_from_embeddings(row["english_variants"], variant_embeddings[index]))

    diagnostics = []
    descriptor_index = {row["descriptor"]: index for index, row in enumerate(rows)}
    for left, right in [
        ("tsundere", "blunt"),
        ("tsundere", "harsh"),
        ("tsundere", "hostile"),
        ("tsundere", "cheeky"),
        ("tsundere", "rebellious"),
        ("tsundere", "japanese language teacher"),
        ("hostile", "kind spirit"),
        ("hostile", "rebellious"),
    ]:
        if left in descriptor_index and right in descriptor_index:
            similarity = regularized_pair_similarity(
                variant_embeddings[descriptor_index[left]],
                variant_embeddings[descriptor_index[right]],
            )
            diagnostics.append(
                {
                    "left": left,
                    "right": right,
                    "regularized_similarity": round(similarity, 8),
                    "regularized_distance": round(1.0 - similarity, 8),
                }
            )

    npz_path = output_path.with_suffix(".npz")
    np.savez_compressed(npz_path, variant_embeddings=variant_embeddings)

    write_json(
        output_path,
        {
            "generated_at": utc_now(),
            "matrix_metadata": str(args.matrix_metadata),
            "translation_backend": args.translation_backend,
            "en_ja_model": args.en_ja_model,
            "ja_en_model": args.ja_en_model,
            "en_lang": args.en_lang,
            "ja_lang": args.ja_lang,
            "ja_top": args.ja_top,
            "en_top_per_ja": args.en_top_per_ja,
            "max_new_tokens": args.max_new_tokens,
            "max_variant_words": args.max_variant_words,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "metric_note": "coherence = largest eigenvalue of 4x4 back-translation cosine-similarity matrix / 4; distance = 1 - coherence",
            "pairwise_note": "pairwise regularized similarity = largest singular value of the 4x4 cross-translation cosine-similarity matrix / 4; distance = 1 - similarity",
            "variant_embedding_cache": str(npz_path),
            "diagnostic_pairs": diagnostics,
            "rows": rows,
        },
    )
    print(f"wrote {npz_path}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
