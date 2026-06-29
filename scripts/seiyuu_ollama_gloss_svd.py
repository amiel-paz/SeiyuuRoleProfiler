#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_translation_regularized_descriptor_distance import encode_bge_small, short_phrase_variants
from scripts.seiyuu_local_orthogonal_svd import (
    character_payload,
    component_list,
    orient_pair,
    seiyuu_characters,
    slug,
    unit_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seiyuu-local SVD with Ollama descriptor gloss overlap.")
    parser.add_argument("--seiyuu", required=True)
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_only_personality_traits.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/seiyuu_ollama_gloss_svd"))
    parser.add_argument("--categories", nargs="+", default=["personality", "traits"])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--component-mass", type=float, default=0.99)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-predict", type=int, default=260)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--max-variant-words", type=int, default=4)
    parser.add_argument("--eigenvalue-tolerance-scale", type=float, default=100.0)
    parser.add_argument(
        "--descriptor-centering",
        choices=["none", "mean"],
        default="none",
        help="Center descriptor gloss embeddings before forming G.",
    )
    parser.add_argument(
        "--character-weighting",
        choices=["none", "log1p_favourites", "sqrt_log1p_favourites"],
        default="none",
        help="Multiply character rows before SVD by a normalized popularity weight.",
    )
    parser.add_argument("--force-gloss-cache", action="store_true")
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


def gloss_cache_key(args: argparse.Namespace, descriptor_count: int) -> str:
    return "_".join(
        [
            slug(args.seiyuu),
            slug(args.tags_input.stem),
            slug(args.ollama_model),
            f"d{descriptor_count}",
        ]
    )


def output_cache_key(args: argparse.Namespace, descriptor_count: int) -> str:
    return "_".join(
        [
            gloss_cache_key(args, descriptor_count),
            f"center_{slug(args.descriptor_centering)}",
            f"weight_{slug(args.character_weighting)}",
        ]
    )


def character_weights(rows: list[dict], weighting: str) -> np.ndarray:
    favourites = np.asarray([float(row.get("favourites") or 0.0) for row in rows], dtype=np.float64)
    if weighting == "none":
        weights = np.ones_like(favourites)
    elif weighting == "log1p_favourites":
        weights = np.log1p(favourites)
    elif weighting == "sqrt_log1p_favourites":
        weights = np.sqrt(np.log1p(favourites))
    else:
        raise RuntimeError(f"Unknown character weighting: {weighting}")
    positive = weights[weights > 0]
    if len(positive):
        weights = weights / positive.mean()
    return weights


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def ollama_generate(args: argparse.Namespace, prompt: str) -> tuple[dict, str]:
    request = urllib.request.Request(
        f"{args.ollama_url.rstrip('/')}/api/generate",
        data=json.dumps(
            {
                "model": args.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": args.temperature,
                    "top_p": 1,
                    "seed": args.seed,
                    "num_predict": args.num_predict,
                    "num_ctx": args.num_ctx,
                },
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = (payload.get("response") or payload.get("thinking") or "").strip()
    return payload, text


def prompt_for_descriptor(descriptor: str) -> str:
    return f"""/no_think
You are normalizing anime character descriptors for embedding.
Descriptor: {descriptor}
Return only JSON with this schema:
{{"descriptor":"{descriptor}","glosses":["phrase1","phrase2","phrase3","phrase4"]}}
Rules:
- Each gloss is English, 1 to 4 words.
- No full sentences.
- No pronouns.
- No punctuation.
- Preserve meaning.
- For anime slang, paraphrase the archetype using ordinary English.
- For job or role descriptors, keep the job or role meaning.
"""


def load_or_create_glosses(args: argparse.Namespace, descriptors: list[str], cache_base: Path) -> tuple[list[dict], np.ndarray]:
    json_path = cache_base.with_name(f"{cache_base.name}_ollama_glosses.json")
    npz_path = cache_base.with_name(f"{cache_base.name}_ollama_glosses.npz")
    if json_path.exists() and npz_path.exists() and not args.force_gloss_cache:
        payload = read_json(json_path)
        if payload.get("descriptors") == descriptors:
            return payload["rows"], np.load(npz_path)["variant_embeddings"].astype(np.float64)

    rows = []
    variant_texts = []
    for index, descriptor in enumerate(descriptors, 1):
        raw_payload, text = ollama_generate(args, prompt_for_descriptor(descriptor))
        try:
            parsed = extract_json(text)
            raw_glosses = [str(value) for value in parsed.get("glosses", [])]
        except Exception:
            raw_glosses = []
        glosses = short_phrase_variants(raw_glosses, descriptor, args.max_variant_words)
        padded = glosses[:4]
        while len(padded) < 4:
            padded.append(padded[-1] if padded else descriptor)
        rows.append(
            {
                "descriptor": descriptor,
                "glosses": padded,
                "raw_text": text,
                "done_reason": raw_payload.get("done_reason"),
                "eval_count": raw_payload.get("eval_count"),
            }
        )
        variant_texts.extend(padded)
        if index % 10 == 0 or index == len(descriptors):
            print(f"glossed {index}/{len(descriptors)}", flush=True)

    variant_embeddings = encode_bge_small(variant_texts).reshape(len(rows), 4, -1).astype(np.float64)
    cache_base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, variant_embeddings=variant_embeddings)
    write_json(
        json_path,
        {
            "generated_at": utc_now(),
            "descriptors": descriptors,
            "ollama_url": args.ollama_url,
            "ollama_model": args.ollama_model,
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "rows": rows,
        },
    )
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    return rows, variant_embeddings


def gloss_overlap(variant_embeddings: np.ndarray) -> np.ndarray:
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


def descriptor_embedding_gram(variant_embeddings: np.ndarray, centering: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = variant_embeddings.mean(axis=1)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    if centering == "none":
        E_used = E
    elif centering == "mean":
        E_used = E - E.mean(axis=0, keepdims=True)
    else:
        raise RuntimeError(f"Unknown descriptor centering: {centering}")
    G = E_used @ E_used.T
    G = (G + G.T) * 0.5
    return E, E_used, G


def pole_representatives(
    B: np.ndarray,
    descriptor_unit: np.ndarray,
    descriptors: list[str],
    rows: list[dict],
    weights: np.ndarray,
    sign: int,
    limit: int = 12,
) -> list[dict]:
    pole = np.maximum(sign * descriptor_unit, 0.0)
    opposing = np.maximum(-sign * descriptor_unit, 0.0)
    output = []
    for row_index, character in enumerate(rows):
        mask = B[row_index] > 0
        pole_sum = float(B[row_index] @ pole)
        opposing_sum = float(B[row_index] @ opposing)
        signed_sum = float(B[row_index] @ descriptor_unit)
        support_indices = np.flatnonzero(mask & (sign * descriptor_unit > 0))
        opposing_indices = np.flatnonzero(mask & (sign * descriptor_unit < 0))
        support = int(len(support_indices))
        total = int(mask.sum())
        if support == 0:
            continue
        purity = pole_sum / (pole_sum + opposing_sum) if (pole_sum + opposing_sum) > 0 else 0.0
        breadth_score = pole_sum * np.sqrt(support)
        representative_score = breadth_score * purity
        support_descriptors = sorted(
            [
                {
                    "descriptor": descriptors[int(index)],
                    "amplitude": round(float(descriptor_unit[int(index)]), 10),
                    "abs_amplitude": round(abs(float(descriptor_unit[int(index)])), 10),
                }
                for index in support_indices
            ],
            key=lambda row: row["abs_amplitude"],
            reverse=True,
        )
        opposing_descriptors = sorted(
            [
                {
                    "descriptor": descriptors[int(index)],
                    "amplitude": round(float(descriptor_unit[int(index)]), 10),
                    "abs_amplitude": round(abs(float(descriptor_unit[int(index)])), 10),
                }
                for index in opposing_indices
            ],
            key=lambda row: row["abs_amplitude"],
            reverse=True,
        )
        output.append(
            {
                **character_payload(character),
                "character_weight": round(float(weights[row_index]), 10),
                "representative_score": round(float(representative_score), 10),
                "breadth_score": round(float(breadth_score), 10),
                "pole_sum": round(float(pole_sum), 10),
                "opposing_sum": round(float(opposing_sum), 10),
                "signed_descriptor_sum": round(float(signed_sum), 10),
                "support": support,
                "opposing_support": int(len(opposing_indices)),
                "total_descriptors": total,
                "purity": round(float(purity), 10),
                "support_descriptors": support_descriptors,
                "opposing_descriptors": opposing_descriptors,
            }
        )
    return sorted(output, key=lambda row: row["representative_score"], reverse=True)[:limit]


def signed_character_components(values: np.ndarray, rows: list[dict], weights: np.ndarray) -> dict[str, list[dict]]:
    positive = []
    negative = []
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        payload = {
            **character_payload(rows[index], float(value)),
            "character_weight": round(float(weights[index]), 10),
        }
        if value >= 0:
            positive.append(payload)
        else:
            negative.append(payload)
    positive.sort(key=lambda row: row["amplitude"], reverse=True)
    negative.sort(key=lambda row: row["amplitude"])
    return {"positive": positive, "negative": negative}


def signed_descriptor_components(values: np.ndarray, descriptors: list[str]) -> dict[str, list[dict]]:
    positive = []
    negative = []
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        payload = {
            "index": int(index),
            "label": descriptors[index],
            "amplitude": round(float(value), 10),
            "abs_amplitude": round(abs(float(value)), 10),
        }
        if value >= 0:
            positive.append(payload)
        else:
            negative.append(payload)
    positive.sort(key=lambda row: row["amplitude"], reverse=True)
    negative.sort(key=lambda row: row["amplitude"])
    return {"positive": positive, "negative": negative}


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
    gloss_base = args.output_dir / gloss_cache_key(args, len(descriptors))
    output_base = args.output_dir / output_cache_key(args, len(descriptors))
    gloss_rows, variant_embeddings = load_or_create_glosses(args, descriptors, gloss_base)
    if args.descriptor_centering == "none":
        E = None
        E_used = None
        G = gloss_overlap(variant_embeddings)
        G_note = "largest singular value of 4x4 cross-gloss BGE-small cosine similarity / 4; diagonal set to 1"
    else:
        E, E_used, G = descriptor_embedding_gram(variant_embeddings, args.descriptor_centering)
        G_note = "centered descriptor gloss Gram: E = normalized mean of 4 gloss embeddings; E_centered = E - mean(E); G = E_centered @ E_centered.T"

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
    M_unweighted = B @ G @ X
    weights = character_weights(rows, args.character_weighting)
    M = M_unweighted * weights[:, None]
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
                        "character_weight": round(float(weights[component["index"]]), 10),
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
                "signed_character_components": signed_character_components(u, rows, weights),
                "signed_descriptor_population_unit_components": signed_descriptor_components(
                    population_descriptor_unit,
                    descriptors,
                ),
                "signed_descriptor_dual_unit_components": signed_descriptor_components(
                    dual_descriptor_unit,
                    descriptors,
                ),
                "positive_pole_representatives": pole_representatives(
                    B,
                    population_descriptor_unit,
                    descriptors,
                    rows,
                    weights,
                    sign=1,
                ),
                "negative_pole_representatives": pole_representatives(
                    B,
                    population_descriptor_unit,
                    descriptors,
                    rows,
                    weights,
                    sign=-1,
                ),
            }
        )

    payload = {
        "generated_at": utc_now(),
        "seiyuu": args.seiyuu,
        "source": "seiyuu_ollama_gloss_svd.py",
        "parameters": {
            "tags_input": str(args.tags_input),
            "categories": sorted(categories),
            "B": "seiyuu character x local descriptor binary incidence matrix",
            "G": "local descriptor x descriptor Ollama-gloss overlap matrix",
            "G_entry": G_note,
            "X": "Lowdin orthogonalizer over positive eigenspace of G: U @ Lambda^(-1/2)",
            "M_unweighted": "B @ G @ X",
            "M": "diag(character_weight) @ B @ G @ X",
            "component_mass": args.component_mass,
            "eigenvalue_tolerance": tolerance,
            "ollama_model": args.ollama_model,
            "temperature": args.temperature,
            "seed": args.seed,
            "descriptor_centering": args.descriptor_centering,
            "character_weighting": args.character_weighting,
            "character_weighting_note": "weights are normalized by mean positive weight so the average positive-weight character remains near 1",
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
        "character_weights": [
            {
                "character_id": int(row["anilist_character_id"]),
                "name": row.get("name") or "",
                "favourites": int(row.get("favourites") or 0),
                "weight": round(float(weight), 10),
            }
            for row, weight in zip(rows, weights)
        ],
        "G_eigenvalues_top20": [round(float(value), 10) for value in eigenvalues[:20]],
        "G_eigenvalues_bottom20": [round(float(value), 10) for value in eigenvalues[-20:]],
        "singular_values": [round(float(value), 10) for value in singular_values],
        "pairs": pairs,
        "characters": [character_payload(character) for character in rows],
        "descriptors": descriptors,
        "gloss_rows": gloss_rows,
    }
    write_json(output_base.with_suffix(".json"), payload)
    np.savez_compressed(
        output_base.with_suffix(".npz"),
        B=B,
        character_weights=weights,
        G=G,
        X=X,
        M_unweighted=M_unweighted,
        M=M,
        G_eigenvalues=eigenvalues,
        G_eigenvectors=eigenvectors,
        G_eigen_keep=keep,
        singular_values=singular_values,
        left_singular_vectors=left,
        right_singular_vectors=right,
    )
    if E is not None and E_used is not None:
        with np.load(output_base.with_suffix(".npz")) as existing:
            values = {key: existing[key] for key in existing.files}
        values["E"] = E
        values["E_used"] = E_used
        np.savez_compressed(output_base.with_suffix(".npz"), **values)
    print(f"wrote {output_base.with_suffix('.json')}")
    print(f"wrote {output_base.with_suffix('.npz')}")
    print(json.dumps({"counts": payload["counts"], "singular_values": payload["singular_values"][:8]}, indent=2))


if __name__ == "__main__":
    main()
