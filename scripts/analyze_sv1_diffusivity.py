#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_adjectival_personality_union import adjectival_canonical, normalize_tag, tokens_for  # noqa: E402
from seiyuu_local_nmf_lane_svd import load_or_create_embeddings  # noqa: E402


DEFAULT_BANGUMI = [
    Path("data/external/safe_enrichment_llm/character_tags.jsonl"),
    Path(
        "run/gpu_llm_tagging/a100_batch_transformers_prod_20260701/"
        "batch_transformers_prod_complete/character_tags_deduped_aggressive.jsonl"
    ),
    Path("run/gpu_llm_tagging/returned_latest/character_tags.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze seiyuu SV1 compactness/diffusivity in the final descriptor basis.")
    parser.add_argument(
        "--basis",
        type=Path,
        default=Path("run/production_personality_basis/final_personality_basis_kept_20260704.tsv"),
    )
    parser.add_argument(
        "--anilist-tags",
        type=Path,
        default=Path("data/external/llm/all_character_description_tags_canonical.json"),
    )
    parser.add_argument("--bangumi-tags", nargs="*", type=Path, default=DEFAULT_BANGUMI)
    parser.add_argument("--output", type=Path, default=Path("run/production_personality_basis/sv1_diffusivity_report.json"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--fit-target", type=float, default=0.95)
    parser.add_argument("--max-fit-terms", type=int, default=30)
    parser.add_argument("seiyuu", nargs="*", default=["Ayana Taketatsu", "Ai Kayano", "Kana Hanazawa"])
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_basis(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["descriptor"] for row in csv.DictReader(handle, dialect="excel-tab")]


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def name_keys(value: str) -> set[str]:
    normalized = norm_name(value)
    parts = [part for part in normalized.split() if part]
    keys = {normalized}
    if len(parts) >= 2:
        keys.add(" ".join(reversed(parts)))
    return {key for key in keys if key}


def tag_values_from_entry(entry: dict) -> list[str]:
    values = []
    for category in ("role", "personality", "traits"):
        for tag in (entry.get("llm_tags") or entry.get("tags") or {}).get(category) or []:
            value = str(tag.get("tag") or "").strip()
            if value:
                values.append(value)
    return values


def candidate_descriptor_values(value: str) -> list[str]:
    normalized = normalize_tag(value)
    values = [normalized]
    values.extend(part.strip() for part in re.split(r"\b(?:and|or|but)\b|[,;/]", normalized) if part.strip())
    values.extend(tokens_for(normalized))
    output = []
    for candidate in values:
        canonical, _ = adjectival_canonical(candidate, 4)
        if canonical:
            output.append(canonical)
    return list(dict.fromkeys(output))


def load_bangumi_tags(paths: list[Path]) -> dict[int, list[str]]:
    rows: dict[int, list[str]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                character_id = payload.get("anilist_character_id")
                if character_id is None:
                    continue
                rows[int(character_id)].extend(tag_values_from_entry(payload))
    return rows


def load_characters(anilist_path: Path, bangumi_paths: list[Path], basis_set: set[str]) -> list[dict]:
    payload = read_json(anilist_path)
    bangumi_by_id = load_bangumi_tags(bangumi_paths)
    characters = []
    for character in payload.get("characters") or []:
        character_id = int(character["anilist_character_id"])
        descriptors: dict[str, float] = {}
        raw_values = tag_values_from_entry(character) + bangumi_by_id.get(character_id, [])
        for raw_value in raw_values:
            for descriptor in candidate_descriptor_values(raw_value):
                if descriptor in basis_set:
                    descriptors[descriptor] = 1.0
        characters.append(
            {
                "character_id": character_id,
                "name": character.get("name") or "",
                "first_anime": character.get("first_anime") or "",
                "favourites": int(character.get("favourites") or 0),
                "site_url": character.get("site_url") or "",
                "image": character.get("image") or "",
                "seiyuu": character.get("seiyuu") or [],
                "descriptors": dict(sorted(descriptors.items())),
            }
        )
    return characters


def lowdin_from_global_gram(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    E = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)
    G = E @ E.T
    G = (G + G.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = max(G.shape) * np.finfo(np.float64).eps * max(float(eigenvalues[0]), 1.0) * 100.0
    keep = eigenvalues > tolerance
    U = eigenvectors[:, keep]
    kept_eigenvalues = eigenvalues[keep]
    X = U @ np.diag(1.0 / np.sqrt(kept_eigenvalues))
    atoms = G @ X
    atoms = atoms / np.maximum(np.linalg.norm(atoms, axis=1, keepdims=True), 1.0e-12)
    return G, X, atoms, E


def nnls_pg(atoms: np.ndarray, target: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    if atoms.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    gram = atoms @ atoms.T
    rhs = atoms @ target
    vector = np.ones(atoms.shape[0], dtype=np.float64)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
    for _ in range(40):
        vector = gram @ vector
        vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
    lipschitz = max(float(vector @ (gram @ vector)), 1.0e-12)
    coefficients = np.zeros(atoms.shape[0], dtype=np.float64)
    y = coefficients.copy()
    t = 1.0
    for _ in range(max_iter):
        next_coefficients = np.maximum(0.0, y - (gram @ y - rhs) / lipschitz)
        next_t = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = next_coefficients + ((t - 1.0) / next_t) * (next_coefficients - coefficients)
        if np.linalg.norm(next_coefficients - coefficients) < 1.0e-10 * max(1.0, np.linalg.norm(coefficients)):
            coefficients = next_coefficients
            break
        coefficients = next_coefficients
        t = next_t
    return coefficients


def greedy_positive_fit(
    target: np.ndarray,
    atoms: np.ndarray,
    descriptors: list[str],
    embeddings: np.ndarray,
    fit_target: float,
    max_terms: int,
    candidate_indices: np.ndarray | None = None,
) -> dict:
    target = target / max(float(np.linalg.norm(target)), 1.0e-12)
    if candidate_indices is None:
        candidate_indices = np.arange(len(descriptors), dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_atoms = atoms[candidate_indices]
    active: list[int] = []
    active_set: set[int] = set()
    residual = target.copy()
    coefficients = np.zeros(0, dtype=np.float64)
    fit = 0.0
    for _ in range(max_terms):
        correlations = candidate_atoms @ residual
        if active_set:
            masked = [np.where(candidate_indices == index)[0][0] for index in active_set]
            correlations[masked] = -np.inf
        candidate_offset = int(np.argmax(correlations))
        if not np.isfinite(correlations[candidate_offset]) or correlations[candidate_offset] <= 0.0:
            break
        index = int(candidate_indices[candidate_offset])
        active.append(index)
        active_set.add(index)
        coefficients = nnls_pg(atoms[active], target)
        approximation = coefficients @ atoms[active]
        fit = float((approximation @ target) / max(float(np.linalg.norm(approximation)), 1.0e-12))
        residual = target - approximation
        if fit >= fit_target:
            break
    if not active:
        return {"fit_percent": 0.0, "descriptors": [], "semantic_diffusivity": None}

    total = max(float(np.sum(coefficients)), 1.0e-12)
    probabilities = coefficients / total
    selected_embeddings = embeddings[np.asarray(active, dtype=np.int64)]
    selected_overlap = selected_embeddings @ selected_embeddings.T
    weighted_semantic_overlap = float(probabilities @ selected_overlap @ probabilities)
    # Half the expected pairwise squared distance:
    #   1/2 E_ij[||e_i - e_j||^2] = E_i[||e_i - mu||^2] = 1 - p.T S p
    # for unit descriptor embeddings and weighted semantic centroid mu.
    semantic_diffusivity = float(max(0.0, 1.0 - weighted_semantic_overlap))
    participation = float(1.0 / max(np.sum(probabilities * probabilities), 1.0e-12))
    pair_rows = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            weight = float(probabilities[i] * probabilities[j])
            if weight <= 0.0:
                continue
            pair_rows.append(
                {
                    "a": descriptors[active[i]],
                    "b": descriptors[active[j]],
                    "overlap": round(float(selected_overlap[i, j]), 6),
                    "half_squared_distance": round(float(1.0 - selected_overlap[i, j]), 6),
                    "pair_probability": round(weight, 8),
                    "diffusivity_contribution": round(float(2.0 * weight * (1.0 - selected_overlap[i, j])), 8),
                }
            )
    pair_rows.sort(key=lambda row: row["diffusivity_contribution"], reverse=True)
    rows = []
    for order_index in np.argsort(coefficients)[::-1]:
        coefficient = float(coefficients[int(order_index)])
        if coefficient <= 1.0e-9:
            continue
        descriptor_index = active[int(order_index)]
        rows.append(
            {
                "descriptor": descriptors[descriptor_index],
                "coefficient": round(coefficient, 8),
                "share_percent": round(float(coefficient / total * 100.0), 4),
            }
        )
    return {
        "fit_percent": round(fit * 100.0, 4),
        "semantic_diffusivity": round(semantic_diffusivity, 6),
        "weighted_semantic_overlap": round(weighted_semantic_overlap, 6),
        "effective_fit_descriptor_count": round(participation, 3),
        "descriptors": rows,
        "largest_pairwise_diffusivity_terms": pair_rows[:10],
    }


def descriptor_distribution_diffusivity(
    weights: np.ndarray,
    descriptors: list[str],
    embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    top_n: int = 20,
) -> dict:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_weights = np.asarray(weights[candidate_indices], dtype=np.float64)
    candidate_weights = np.maximum(candidate_weights, 0.0)
    total = float(np.sum(candidate_weights))
    if total <= 1.0e-12:
        return {
            "semantic_diffusivity": None,
            "weighted_semantic_overlap": None,
            "effective_descriptor_count": 0.0,
            "descriptors": [],
            "largest_pairwise_diffusivity_terms": [],
        }
    probabilities = candidate_weights / total
    selected_embeddings = embeddings[candidate_indices]
    selected_overlap = selected_embeddings @ selected_embeddings.T
    weighted_semantic_overlap = float(probabilities @ selected_overlap @ probabilities)
    semantic_diffusivity = float(max(0.0, 1.0 - weighted_semantic_overlap))
    participation = float(1.0 / max(np.sum(probabilities * probabilities), 1.0e-12))
    order = np.argsort(probabilities)[::-1]
    rows = [
        {
            "descriptor": descriptors[int(candidate_indices[index])],
            "share_percent": round(float(probabilities[index] * 100.0), 4),
            "loading": round(float(math.sqrt(candidate_weights[index])), 6),
        }
        for index in order[:top_n]
        if probabilities[index] > 0.0
    ]
    active = order[: min(30, len(order))]
    pair_rows = []
    for offset_i, i in enumerate(active):
        for j in active[offset_i + 1 :]:
            weight = float(probabilities[i] * probabilities[j])
            if weight <= 0.0:
                continue
            pair_rows.append(
                {
                    "a": descriptors[int(candidate_indices[i])],
                    "b": descriptors[int(candidate_indices[j])],
                    "overlap": round(float(selected_overlap[i, j]), 6),
                    "half_squared_distance": round(float(1.0 - selected_overlap[i, j]), 6),
                    "pair_probability": round(weight, 8),
                    "diffusivity_contribution": round(float(2.0 * weight * (1.0 - selected_overlap[i, j])), 8),
                }
            )
    pair_rows.sort(key=lambda row: row["diffusivity_contribution"], reverse=True)
    return {
        "semantic_diffusivity": round(semantic_diffusivity, 6),
        "weighted_semantic_overlap": round(weighted_semantic_overlap, 6),
        "effective_descriptor_count": round(participation, 3),
        "descriptors": rows,
        "largest_pairwise_diffusivity_terms": pair_rows[:10],
    }


def analyze_seiyuu(
    name: str,
    characters: list[dict],
    descriptors: list[str],
    descriptor_index: dict[str, int],
    G: np.ndarray,
    X: np.ndarray,
    atoms: np.ndarray,
    embeddings: np.ndarray,
    fit_target: float,
    max_fit_terms: int,
) -> dict:
    requested = name_keys(name)
    rows = []
    for character in characters:
        if not any(requested.intersection(name_keys(row.get("name") or "")) for row in character.get("seiyuu") or []):
            continue
        if character["descriptors"]:
            rows.append(character)
    B = np.zeros((len(rows), len(descriptors)), dtype=np.float64)
    for row_index, character in enumerate(rows):
        for descriptor, weight in character["descriptors"].items():
            if descriptor in descriptor_index:
                B[row_index, descriptor_index[descriptor]] = float(weight)
    M = B @ G @ X
    if M.shape[0] == 0:
        raise RuntimeError(f"No descriptor-bearing characters for {name}")
    left, singular_values, vt = np.linalg.svd(M, full_matrices=False)
    v1 = vt[0].copy()
    u1 = left[:, 0].copy()
    loading = atoms @ v1
    if loading[int(np.argmax(np.abs(loading)))] < 0:
        v1 *= -1.0
        u1 *= -1.0
        loading *= -1.0
    supported_descriptor_indices = np.flatnonzero(B.sum(axis=0) > 0.0)
    fit = greedy_positive_fit(
        v1,
        atoms,
        descriptors,
        embeddings,
        fit_target,
        max_fit_terms,
        candidate_indices=supported_descriptor_indices,
    )
    sv1_supported_loading_distribution = descriptor_distribution_diffusivity(
        loading * loading,
        descriptors,
        embeddings,
        supported_descriptor_indices,
    )
    sv1_global_loading_distribution = descriptor_distribution_diffusivity(
        loading * loading,
        descriptors,
        embeddings,
        np.arange(len(descriptors), dtype=np.int64),
    )
    sv_mass = singular_values * singular_values
    char_norm = max(float(np.sum(u1 * u1)), 1.0e-12)
    character_rows = []
    for index in np.argsort(u1 * u1)[::-1][:10]:
        character = rows[int(index)]
        character_rows.append(
            {
                "name": character["name"],
                "anime": character["first_anime"],
                "amplitude": round(float(u1[int(index)]), 6),
                "share_percent": round(float(u1[int(index)] ** 2 / char_norm * 100.0), 4),
                "favourites": character["favourites"],
                "descriptors": sorted(character["descriptors"])[:20],
            }
        )
    return {
        "seiyuu": name,
        "character_count_with_basis_descriptors": len(rows),
        "basis_descriptor_count": len(descriptors),
        "global_G_shape": list(G.shape),
        "global_X_shape": list(X.shape),
        "matrix": "B @ G @ X with G and X built once from the full final kept basis",
        "singular_value_1": round(float(singular_values[0]), 6),
        "sv1_mass_percent": round(float(sv_mass[0] / max(np.sum(sv_mass), 1.0e-12) * 100.0), 4),
        "sv1_supported_descriptor_fit": fit,
        "sv1_supported_loading_distribution": sv1_supported_loading_distribution,
        "sv1_global_loading_distribution": sv1_global_loading_distribution,
        "top_characters_by_sv1": character_rows,
    }


def main() -> None:
    args = parse_args()
    descriptors = read_basis(args.basis)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    embeddings = load_or_create_embeddings(descriptors, args.output.parent, args.embedding_model).astype(np.float64)
    G, X, atoms, E = lowdin_from_global_gram(embeddings)
    characters = load_characters(args.anilist_tags, args.bangumi_tags, set(descriptors))
    payload = {
        "basis": str(args.basis),
        "embedding_model": args.embedding_model,
        "fit_target": args.fit_target,
        "notes": {
            "X": "Lowdin/symmetric orthogonalizer from eigendecomposition of the full final-basis Gram matrix G=E@E.T.",
            "semantic_diffusivity": "Pairwise semantic spread from direct SV1 descriptor loadings, not compact fit descriptors: D = 1 - p.T S p = 1/2 sum_ij p_i p_j ||e_i-e_j||^2, where S_ij is descriptor embedding cosine overlap and p_i is normalized squared loading.",
            "descriptor_fit": "The compact positive descriptor fit is restricted to descriptors with direct support on that seiyuu's characters.",
            "B": "Plain binary character x final-basis descriptor incidence; no popularity weighting and no row normalization.",
        },
        "G_shape": list(G.shape),
        "X_shape": list(X.shape),
        "results": [
            analyze_seiyuu(name, characters, descriptors, descriptor_index, G, X, atoms, E, args.fit_target, args.max_fit_terms)
            for name in args.seiyuu
        ],
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
