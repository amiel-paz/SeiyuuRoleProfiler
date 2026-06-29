#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_translation_regularized_descriptor_distance import (
    encode_bge_small,
    load_translation,
    short_phrase_variants,
    translate_batch,
)
from scripts.seiyuu_local_orthogonal_svd import (
    character_descriptors,
    character_payload,
    component_list,
    name_keys,
    orient_pair,
    seiyuu_characters,
    slug,
    unit_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seiyuu-local SVD with translation-regularized descriptor overlap.")
    parser.add_argument("--seiyuu", required=True)
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_only_personality_traits.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/seiyuu_translation_regularized_svd"))
    parser.add_argument("--categories", nargs="+", default=["personality", "traits"])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--component-mass", type=float, default=0.99)
    parser.add_argument("--translation-backend", choices=["nllb", "marian"], default="nllb")
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
    parser.add_argument("--eigenvalue-tolerance-scale", type=float, default=100.0)
    parser.add_argument("--force-translation-cache", action="store_true")
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


def local_characters(tag_payload: dict, seiyuu: str, categories: set[str]) -> tuple[list[dict], list[dict]]:
    all_rows = seiyuu_characters(tag_payload, seiyuu, categories)
    rows = [row for row in all_rows if row["_descriptors"]]
    if not rows:
        raise RuntimeError(f"No descriptor-bearing characters found for {seiyuu!r}.")
    return all_rows, rows


def cache_key(args: argparse.Namespace, descriptor_count: int) -> str:
    return "_".join(
        [
            slug(args.seiyuu),
            slug(args.tags_input.stem),
            slug(args.translation_backend),
            slug(args.en_ja_model),
            slug(args.ja_en_model),
            f"d{descriptor_count}",
        ]
    )


def load_or_create_translation_variants(
    args: argparse.Namespace,
    descriptors: list[str],
    cache_base: Path,
) -> tuple[list[dict], np.ndarray]:
    json_path = cache_base.with_name(f"{cache_base.name}_translation_variants.json")
    npz_path = cache_base.with_name(f"{cache_base.name}_translation_variants.npz")
    if json_path.exists() and npz_path.exists() and not args.force_translation_cache:
        payload = read_json(json_path)
        if payload.get("descriptors") == descriptors:
            return payload["rows"], np.load(npz_path)["variant_embeddings"].astype(np.float64)

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

    variant_embeddings = encode_bge_small(variant_texts).reshape(len(rows), 4, -1).astype(np.float64)
    cache_base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, variant_embeddings=variant_embeddings)
    write_json(
        json_path,
        {
            "generated_at": utc_now(),
            "descriptors": descriptors,
            "translation_backend": args.translation_backend,
            "en_ja_model": args.en_ja_model,
            "ja_en_model": args.ja_en_model,
            "en_lang": args.en_lang,
            "ja_lang": args.ja_lang,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "max_new_tokens": args.max_new_tokens,
            "max_variant_words": args.max_variant_words,
            "rows": rows,
        },
    )
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    return rows, variant_embeddings


def translation_overlap(variant_embeddings: np.ndarray) -> np.ndarray:
    n = variant_embeddings.shape[0]
    G = np.zeros((n, n), dtype=np.float64)
    for left in range(n):
        for right in range(left, n):
            cross_similarity = variant_embeddings[left] @ variant_embeddings[right].T
            similarity = float(np.linalg.svd(cross_similarity, compute_uv=False)[0] / variant_embeddings.shape[1])
            G[left, right] = similarity
            G[right, left] = similarity
    np.fill_diagonal(G, 1.0)
    return (G + G.T) * 0.5


def main() -> None:
    args = parse_args()
    categories = {category.strip() for category in args.categories}
    tag_payload = read_json(args.tags_input)
    all_characters, rows = local_characters(tag_payload, args.seiyuu, categories)

    descriptors = sorted({descriptor for character in rows for descriptor in character["_descriptors"]})
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    B = np.zeros((len(rows), len(descriptors)), dtype=np.float64)
    for row_index, character in enumerate(rows):
        for descriptor in character["_descriptors"]:
            B[row_index, descriptor_index[descriptor]] = 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / cache_key(args, len(descriptors))
    translation_rows, variant_embeddings = load_or_create_translation_variants(args, descriptors, output_base)
    G = translation_overlap(variant_embeddings)

    eigenvalues, eigenvectors = np.linalg.eigh(G)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = (
        max(G.shape) * np.finfo(np.float64).eps * max(float(eigenvalues[0]), 1.0) * args.eigenvalue_tolerance_scale
    )
    keep = eigenvalues > tolerance
    retained_eigenvalues = eigenvalues[keep]
    retained_eigenvectors = eigenvectors[:, keep]
    X = retained_eigenvectors @ np.diag(1.0 / np.sqrt(retained_eigenvalues))
    M = B @ G @ X
    left, singular_values, vt = np.linalg.svd(M, full_matrices=False)
    right = vt.T

    pairs = []
    for pair_index in range(min(args.pairs, len(singular_values))):
        u, v = orient_pair(left[:, pair_index].copy(), right[:, pair_index].copy(), G, X)
        population_descriptor_loading = G @ X @ v
        dual_descriptor_loading = X @ v
        population_descriptor_unit, population_descriptor_norm = unit_vector(population_descriptor_loading)
        dual_descriptor_unit, dual_descriptor_norm = unit_vector(dual_descriptor_loading)
        character_components = component_list(u, [character["name"] for character in rows], args.component_mass)
        pairs.append(
            {
                "rank": pair_index + 1,
                "singular_value": round(float(singular_values[pair_index]), 10),
                "vector_norms": {
                    "left_character_singular_vector": round(float(np.linalg.norm(u)), 10),
                    "right_orthogonal_singular_vector": round(float(np.linalg.norm(v)), 10),
                    "descriptor_population_projection_raw": round(population_descriptor_norm, 10),
                    "descriptor_dual_projection_raw": round(dual_descriptor_norm, 10),
                },
                "top_character_components": [
                    {
                        **character_payload(rows[component["index"]], component["amplitude"]),
                        "l2_mass_share": component["l2_mass_share"],
                        "cumulative_l2_mass": component["cumulative_l2_mass"],
                    }
                    for component in character_components
                ],
                "top_descriptor_population_unit_components": component_list(
                    population_descriptor_unit,
                    descriptors,
                    args.component_mass,
                ),
                "top_descriptor_dual_unit_components": component_list(
                    dual_descriptor_unit,
                    descriptors,
                    args.component_mass,
                ),
            }
        )

    payload = {
        "generated_at": utc_now(),
        "seiyuu": args.seiyuu,
        "source": "seiyuu_translation_regularized_svd.py",
        "parameters": {
            "tags_input": str(args.tags_input),
            "categories": sorted(categories),
            "B": "seiyuu character x local descriptor binary incidence matrix",
            "G": "local descriptor x descriptor translation-regularized overlap matrix",
            "G_entry": "largest singular value of 4x4 cross-translation BGE-small cosine similarity / 4; diagonal set to 1",
            "X": "Lowdin orthogonalizer over positive eigenspace of G: U @ Lambda^(-1/2)",
            "M": "B @ G @ X",
            "component_mass": args.component_mass,
            "eigenvalue_tolerance": tolerance,
        },
        "counts": {
            "seiyuu_characters_total": len(all_characters),
            "characters_with_descriptors": len(rows),
            "local_descriptors": len(descriptors),
            "binary_nonzero_entries": int(B.sum()),
            "G_rank_retained": int(keep.sum()),
            "G_rank_dropped": int((~keep).sum()),
            "G_negative_eigenvalues": int((eigenvalues < -tolerance).sum()),
        },
        "G_eigenvalues_top20": [round(float(value), 10) for value in eigenvalues[:20]],
        "G_eigenvalues_bottom20": [round(float(value), 10) for value in eigenvalues[-20:]],
        "singular_values": [round(float(value), 10) for value in singular_values],
        "pairs": pairs,
        "characters": [character_payload(character) for character in rows],
        "descriptors": descriptors,
        "translation_rows": translation_rows,
    }
    write_json(output_base.with_suffix(".json"), payload)
    np.savez_compressed(
        output_base.with_suffix(".npz"),
        B=B,
        G=G,
        X=X,
        M=M,
        G_eigenvalues=eigenvalues,
        G_eigenvectors=eigenvectors,
        G_eigen_keep=keep,
        singular_values=singular_values,
        left_singular_vectors=left,
        right_singular_vectors=right,
    )
    print(f"wrote {output_base.with_suffix('.json')}")
    print(f"wrote {output_base.with_suffix('.npz')}")
    print(json.dumps({"counts": payload["counts"], "singular_values": payload["singular_values"][:8]}, indent=2))


if __name__ == "__main__":
    main()
