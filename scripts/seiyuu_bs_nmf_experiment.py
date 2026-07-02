#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import NMF

from build_svd_site import (
    alias_keys,
    character_row_weight,
    descriptor_weights_from_tag_row,
    descriptors_from_tag_row,
    import_nltk,
    is_pop_team_character,
    name_keys,
    pure_adjective_descriptor,
    read_json,
    shared_role_weight,
    slug,
    utc_now,
    write_json,
)
from comparative_personality import DEFAULT_INHERITANCE_WEIGHT, inherited_personality_descriptors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit local nonnegative seiyuu lanes from B @ S, where S smooths into adjectival descriptor space."
    )
    parser.add_argument("seiyuu", help="Romaji seiyuu query, e.g. 'Takehito Koyasu'.")
    parser.add_argument("--site-profile-input", type=Path, default=Path("site/profiles.json"))
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"),
    )
    parser.add_argument(
        "--glosses-json",
        type=Path,
        default=Path(
            "models/global_ollama_descriptor_glosses/"
            "all_characters_llm_vndb_personality_tags_qwen3_5_4b_personality_role_traits_filtered_all_ollama_glosses.json"
        ),
    )
    parser.add_argument(
        "--glosses-npz",
        type=Path,
        default=Path(
            "models/global_ollama_descriptor_glosses/"
            "all_characters_llm_vndb_personality_tags_qwen3_5_4b_personality_role_traits_filtered_all_ollama_glosses.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/seiyuu_bs_nmf"))
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--top-descriptors", type=int, default=18)
    parser.add_argument("--top-characters", type=int, default=12)
    parser.add_argument(
        "--descriptor-scope",
        choices=["no_roles", "kitchen_sink"],
        default="no_roles",
    )
    parser.add_argument(
        "--similarity-mode",
        choices=["positive_cosine", "power_positive_cosine", "threshold_positive_cosine"],
        default="power_positive_cosine",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--similarity-power", type=float, default=3.0)
    parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=0,
        help="If positive, keep only the top-k target descriptors for each source descriptor after thresholding.",
    )
    parser.add_argument("--normalize-s-columns", action="store_true")
    parser.add_argument("--row-normalize-x", action="store_true")
    parser.add_argument(
        "--row-weight",
        choices=["none", "sqrt_log_favourites", "log_favourites", "sqrt_favourites"],
        default="sqrt_log_favourites",
    )
    parser.add_argument(
        "--descriptor-support-weight",
        choices=["none", "linear_cap", "sqrt_cap", "log_cap"],
        default="none",
        help="Confidence weight based on how many descriptors a character has; cap reaches 1 at --descriptor-support-full-count.",
    )
    parser.add_argument("--descriptor-support-full-count", type=int, default=6)
    parser.add_argument(
        "--min-descriptor-count",
        type=int,
        default=1,
        help="Drop characters with fewer retained descriptors before fitting local lanes.",
    )
    parser.add_argument(
        "--shared-role-weight",
        choices=["none", "inverse_sqrt", "inverse"],
        default="inverse_sqrt",
    )
    parser.add_argument("--max-role-edge-count", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    col_sums = matrix.sum(axis=0, keepdims=True)
    return np.divide(matrix, col_sums, out=np.zeros_like(matrix), where=col_sums > 0)


def descriptor_support_weight(character: dict, mode: str, full_count: int) -> float:
    if mode == "none":
        return 1.0
    count = len(set(character.get("descriptors") or []))
    full_count = max(1, full_count)
    if mode == "linear_cap":
        return min(count / full_count, 1.0)
    if mode == "sqrt_cap":
        return min(count / full_count, 1.0) ** 0.5
    if mode == "log_cap":
        return min(np.log1p(count) / np.log1p(full_count), 1.0)
    raise ValueError(f"unknown descriptor support weight: {mode}")


def descriptor_embeddings(glosses_npz: Path) -> np.ndarray:
    variant_embeddings = np.load(glosses_npz)["variant_embeddings"].astype(np.float64)
    embeddings = variant_embeddings.mean(axis=1)
    return embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)


def build_similarity(
    descriptor_embeddings_: np.ndarray,
    source_indices: list[int],
    target_indices: list[int],
    mode: str,
    threshold: float,
    power: float,
    top_k: int,
    normalize_columns_: bool,
) -> np.ndarray:
    source = descriptor_embeddings_[source_indices]
    target = descriptor_embeddings_[target_indices]
    similarity = np.clip(source @ target.T, 0.0, 1.0)
    if mode == "threshold_positive_cosine":
        similarity[similarity < threshold] = 0.0
    elif mode == "power_positive_cosine":
        similarity = np.power(similarity, power)
        similarity[similarity < threshold**power] = 0.0
    elif mode != "positive_cosine":
        raise ValueError(f"unknown similarity mode: {mode}")
    if top_k > 0 and top_k < similarity.shape[1]:
        keep = np.zeros_like(similarity, dtype=bool)
        top_indices = np.argpartition(similarity, -top_k, axis=1)[:, -top_k:]
        row_indices = np.arange(similarity.shape[0])[:, None]
        keep[row_indices, top_indices] = similarity[row_indices, top_indices] > 0.0
        similarity = np.where(keep, similarity, 0.0)
    if normalize_columns_:
        similarity = normalize_columns(similarity)
    return similarity.astype(np.float64)


def find_profile(profiles: list[dict], query: str) -> dict:
    query_keys = name_keys(query)
    for profile in profiles:
        if query_keys & alias_keys(profile):
            return profile
    raise RuntimeError(f"No profile matched query: {query}")


def build_seiyuu_characters(
    profile: dict,
    tag_payload: dict,
    descriptor_index: dict[str, int],
    descriptor_scope: str,
    max_role_edge_count: int,
    min_descriptor_count: int,
) -> list[dict]:
    seiyuu_to_characters: dict[str, list[dict]] = defaultdict(list)
    inherited_by_character_id, _ = inherited_personality_descriptors(
        tag_payload.get("characters", []),
        DEFAULT_INHERITANCE_WEIGHT,
    )
    for source in tag_payload.get("characters", []):
        role_edge_count = len(source.get("seiyuu") or [])
        if role_edge_count > max_role_edge_count or is_pop_team_character(source):
            continue
        descriptor_weights = descriptor_weights_from_tag_row(
            source,
            descriptor_index,
            descriptor_scope,
            inherited_by_character_id,
        )
        descriptors = sorted(descriptor_weights)
        if len(set(descriptors)) < min_descriptor_count:
            continue
        character = {
            "character_id": int(source["anilist_character_id"]),
            "name": source.get("name") or "",
            "first_anime": source.get("first_anime") or "",
            "favourites": int(source.get("favourites") or 0),
            "site_url": source.get("site_url") or "",
            "descriptors": descriptors,
            "descriptor_weights": descriptor_weights,
            "role_edge_count": max(role_edge_count, 1),
        }
        for seiyuu in source.get("seiyuu", []):
            for key in name_keys(seiyuu.get("name") or ""):
                seiyuu_to_characters[key].append(character)

    seen = set()
    output = []
    for key in alias_keys(profile):
        for character in seiyuu_to_characters.get(key, []):
            character_id = int(character["character_id"])
            if character_id in seen:
                continue
            seen.add(character_id)
            output.append(character)
    return output


def build_binary_matrix(
    characters: list[dict],
    descriptor_index: dict[str, int],
    source_indices: list[int],
) -> tuple[np.ndarray, dict[int, int]]:
    source_position = {descriptor_index: position for position, descriptor_index in enumerate(source_indices)}
    matrix = np.zeros((len(characters), len(source_indices)), dtype=np.float64)
    for row_index, character in enumerate(characters):
        for descriptor in character["descriptors"]:
            global_index = descriptor_index.get(descriptor)
            if global_index in source_position:
                matrix[row_index, source_position[global_index]] = float(
                    character.get("descriptor_weights", {}).get(descriptor, 1.0)
                )
    return matrix, source_position


def fit_nmf(matrix: np.ndarray, components: int, random_state: int, max_iter: int) -> tuple[NMF, np.ndarray, np.ndarray]:
    components = max(1, min(components, matrix.shape[0], matrix.shape[1]))
    nmf = NMF(
        n_components=components,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        random_state=random_state,
        max_iter=max_iter,
    )
    W = nmf.fit_transform(matrix)
    H = nmf.components_
    return nmf, W, H


def explained_entries(matrix: np.ndarray, reconstruction: np.ndarray) -> dict:
    residual = matrix - reconstruction
    frobenius = float(np.linalg.norm(matrix))
    residual_norm = float(np.linalg.norm(residual))
    return {
        "frobenius_norm": round(frobenius, 8),
        "residual_norm": round(residual_norm, 8),
        "relative_reconstruction_error": round(residual_norm / frobenius, 8) if frobenius else 0.0,
        "matrix_density": round(float((matrix > 0).sum() / matrix.size), 8) if matrix.size else 0.0,
    }


def character_payload(character: dict, loading: float, row_weight: float) -> dict:
    return {
        "character_id": character["character_id"],
        "name": character["name"],
        "anime": character["first_anime"],
        "favourites": character["favourites"],
        "site_url": character["site_url"],
        "descriptors": character["descriptors"],
        "descriptor_count": len(set(character["descriptors"])),
        "role_edge_count": character["role_edge_count"],
        "row_weight": round(float(row_weight), 8),
        "loading": round(float(loading), 8),
    }


def lane_payloads(
    W: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    characters: list[dict],
    source_descriptors: list[str],
    target_descriptors: list[str],
    row_weights: np.ndarray,
    top_descriptors: int,
    top_characters: int,
) -> list[dict]:
    lane_strengths = W.sum(axis=0)
    total_strength = float(lane_strengths.sum())
    lanes = []
    for lane_index in range(H.shape[0]):
        descriptor_weights = H[lane_index]
        source_descriptor_weights = S @ descriptor_weights
        descriptor_total = float(descriptor_weights.sum())
        source_descriptor_total = float(source_descriptor_weights.sum())
        descriptor_order = np.argsort(descriptor_weights)[::-1]
        source_descriptor_order = np.argsort(source_descriptor_weights)[::-1]
        character_loading = W[:, lane_index]
        character_order = np.argsort(character_loading)[::-1]
        lanes.append(
            {
                "lane": lane_index,
                "strength": round(float(lane_strengths[lane_index]), 8),
                "strength_share": round(float(lane_strengths[lane_index] / total_strength), 8)
                if total_strength
                else 0.0,
                "top_descriptors": [
                    {
                        "descriptor": target_descriptors[int(index)],
                        "weight": round(float(descriptor_weights[int(index)]), 8),
                        "share": round(float(descriptor_weights[int(index)] / descriptor_total), 8)
                        if descriptor_total
                        else 0.0,
                    }
                    for index in descriptor_order[:top_descriptors]
                    if float(descriptor_weights[int(index)]) > 0.0
                ],
                "top_source_descriptors": [
                    {
                        "descriptor": source_descriptors[int(index)],
                        "weight": round(float(source_descriptor_weights[int(index)]), 8),
                        "share": round(float(source_descriptor_weights[int(index)] / source_descriptor_total), 8)
                        if source_descriptor_total
                        else 0.0,
                    }
                    for index in source_descriptor_order[:top_descriptors]
                    if float(source_descriptor_weights[int(index)]) > 0.0
                ],
                "top_characters": [
                    character_payload(characters[int(index)], float(character_loading[int(index)]), float(row_weights[int(index)]))
                    for index in character_order[:top_characters]
                    if float(character_loading[int(index)]) > 0.0
                ],
            }
        )
    lanes.sort(key=lambda lane: lane["strength"], reverse=True)
    return lanes


def main() -> None:
    args = parse_args()
    old_payload = read_json(args.site_profile_input)
    tag_payload = read_json(args.tags_input)
    gloss_payload = read_json(args.glosses_json)
    all_descriptors = [row["descriptor"] for row in gloss_payload["rows"]]
    descriptor_index = {descriptor: index for index, descriptor in enumerate(all_descriptors)}
    profile = find_profile(old_payload["profiles"], args.seiyuu)
    characters = build_seiyuu_characters(
        profile,
        tag_payload,
        descriptor_index,
        args.descriptor_scope,
        args.max_role_edge_count,
        args.min_descriptor_count,
    )
    if len(characters) < 2:
        raise RuntimeError(f"Need at least two described characters for {profile['name']}; found {len(characters)}.")

    nltk, wn = import_nltk()
    target_indices = [
        index for index, descriptor in enumerate(all_descriptors) if pure_adjective_descriptor(descriptor, nltk, wn)
    ]
    source_indices = sorted(
        {
            descriptor_index[descriptor]
            for character in characters
            for descriptor in character["descriptors"]
            if descriptor in descriptor_index
        }
    )
    source_descriptors = [all_descriptors[index] for index in source_indices]
    target_descriptors = [all_descriptors[index] for index in target_indices]
    B, _ = build_binary_matrix(characters, descriptor_index, source_indices)
    embeddings = descriptor_embeddings(args.glosses_npz)
    S = build_similarity(
        embeddings,
        source_indices,
        target_indices,
        args.similarity_mode,
        args.similarity_threshold,
        args.similarity_power,
        args.similarity_top_k,
        args.normalize_s_columns,
    )
    X = B @ S
    row_weights = np.asarray(
        [
            character_row_weight(character, args.row_weight)
            * shared_role_weight(character, args.shared_role_weight)
            * descriptor_support_weight(
                character,
                args.descriptor_support_weight,
                args.descriptor_support_full_count,
            )
            for character in characters
        ],
        dtype=np.float64,
    )
    X_weighted = X.copy()
    if args.row_normalize_x:
        X_weighted = row_normalize(X_weighted)
    X_weighted = X_weighted * row_weights.reshape(-1, 1)
    nmf, W, H = fit_nmf(X_weighted, args.components, args.random_state, args.max_iter)
    reconstruction = W @ H

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{slug(profile['name'])}_bs_nmf_k{H.shape[0]:02d}_"
        f"{args.similarity_mode}_t{str(args.similarity_threshold).replace('.', 'p')}_"
        f"p{str(args.similarity_power).replace('.', 'p')}"
    )
    if args.similarity_top_k > 0:
        stem += f"_topk{args.similarity_top_k}"
    if args.row_normalize_x:
        stem += "_rownorm"
    if args.descriptor_support_weight != "none":
        stem += f"_desc{args.descriptor_support_weight}{args.descriptor_support_full_count}"
    if args.min_descriptor_count > 1:
        stem += f"_mindesc{args.min_descriptor_count}"
    json_path = args.output_dir / f"{stem}.json"
    npz_path = args.output_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, B=B, S=S, X=X, X_weighted=X_weighted, W=W, H=H, reconstruction=reconstruction)
    payload = {
        "generated_at": utc_now(),
        "source": "seiyuu_bs_nmf_experiment.py",
        "parameters": {
            "seiyuu_query": args.seiyuu,
            "profile_name": profile["name"],
            "descriptor_scope": args.descriptor_scope,
            "components": int(H.shape[0]),
            "similarity_mode": args.similarity_mode,
            "similarity_threshold": args.similarity_threshold,
            "similarity_power": args.similarity_power,
            "similarity_top_k": args.similarity_top_k,
            "normalize_s_columns": args.normalize_s_columns,
            "row_normalize_x": args.row_normalize_x,
            "row_weight": args.row_weight,
            "descriptor_support_weight": args.descriptor_support_weight,
            "descriptor_support_full_count": args.descriptor_support_full_count,
            "min_descriptor_count": args.min_descriptor_count,
            "shared_role_weight": args.shared_role_weight,
            "max_role_edge_count": args.max_role_edge_count,
            "comparative_personality": (
                "direct descriptors are 1.0; explicit personality-similarity references to resolvable "
                f"corpus characters inherit target personality descriptors at {DEFAULT_INHERITANCE_WEIGHT}"
            ),
            "random_state": args.random_state,
            "model": "X = B @ S; B is local character x no-role weighted descriptor incidence; S maps descriptors into pure-adjectival descriptor space with nonnegative embedding similarity; local NMF factors X into nonnegative character lanes and descriptor lanes.",
        },
        "counts": {
            "characters": len(characters),
            "source_descriptors": len(source_descriptors),
            "target_adjectival_descriptors": len(target_descriptors),
            "B_nonzero": int((B > 0).sum()),
            "S_nonzero": int((S > 0).sum()),
            "X_nonzero": int((X_weighted > 0).sum()),
        },
        "fit": {
            "n_iter": int(nmf.n_iter_),
            "reconstruction_err": round(float(nmf.reconstruction_err_), 8),
            **explained_entries(X_weighted, reconstruction),
        },
        "source_descriptors": source_descriptors,
        "target_adjectival_descriptors": target_descriptors,
        "characters": [
            character_payload(character, 0.0, float(row_weights[index])) for index, character in enumerate(characters)
        ],
        "lanes": lane_payloads(
            W,
            H,
            S,
            characters,
            source_descriptors,
            target_descriptors,
            row_weights,
            args.top_descriptors,
            args.top_characters,
        ),
        "matrix_npz": str(npz_path),
    }
    write_json(json_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    print(json.dumps({"counts": payload["counts"], "fit": payload["fit"]}, indent=2))


if __name__ == "__main__":
    main()
