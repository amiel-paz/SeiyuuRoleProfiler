#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from seiyuu_local_nmf_lane_svd import (
    alias_keys,
    character_row_weight,
    descriptor_rows_from_character,
    enrich_character,
    find_profile,
    is_pop_team_character,
    load_bangumi_collects,
    load_or_create_embeddings,
    load_role_character_display,
    load_safe_llm_personality,
    name_keys,
    normalize_rows,
    read_json,
    shared_role_weight,
    slug,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster a seiyuu's characters in normalized orthogonal semantic descriptor space."
    )
    parser.add_argument("seiyuu", help="Romaji seiyuu query, e.g. 'Ayana Taketatsu'.")
    parser.add_argument("--site-profile-input", type=Path, default=Path("site/profiles.json"))
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"),
    )
    parser.add_argument(
        "--safe-llm-tags",
        nargs="*",
        type=Path,
        default=[
            Path("data/external/safe_enrichment_llm/character_tags.jsonl"),
            Path("run/gpu_llm_tagging/returned_latest/character_tags.jsonl"),
        ],
    )
    parser.add_argument(
        "--safe-enrichment",
        type=Path,
        default=Path("data/external/safe_enrichment/character_safe_enrichment.jsonl"),
    )
    parser.add_argument(
        "--bangumi-raw-dir",
        type=Path,
        default=Path("data/external/safe_enrichment/raw/bangumi"),
    )
    parser.add_argument("--character-display-input", type=Path, default=Path("site/character_display.json"))
    parser.add_argument("--role-edges", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/seiyuu_character_semantic_clusters"))
    parser.add_argument(
        "--global-canonicalization-input",
        type=Path,
        default=Path("models/global_descriptor_canonicalization/descriptor_canonicalization.json"),
        help="Optional global raw-to-canonical descriptor map. If present, it is used before local fallback canonicalization.",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument("--max-role-edge-count", type=int, default=20)
    parser.add_argument(
        "--canonicalize-similarity-threshold",
        type=float,
        default=1.01,
        help="Merge very similar local descriptors before matrix construction; set >1 to disable.",
    )
    parser.add_argument(
        "--canonicalize-contained-distance-threshold",
        type=float,
        default=0.16,
        help="Merge a longer descriptor into a contained shorter descriptor when their cosine distance is small; set <0 to disable.",
    )
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--n-init", type=int, default=100)
    parser.add_argument("--cluster-method", choices=["recursive", "kmeans-scan"], default="recursive")
    parser.add_argument("--radius-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-split-gain", type=float, default=0.10)
    parser.add_argument("--min-split-distance", type=float, default=0.04)
    parser.add_argument("--max-recursive-depth", type=int, default=4)
    parser.add_argument("--no-trim-outliers", action="store_true")
    parser.add_argument("--descriptor-fit-target", type=float, default=0.95)
    parser.add_argument("--descriptor-fit-min-terms", type=int, default=6)
    parser.add_argument("--descriptor-fit-max-terms", type=int, default=12)
    parser.add_argument("--top-descriptors", type=int, default=16)
    parser.add_argument("--top-characters", type=int, default=16)
    parser.add_argument(
        "--row-weight",
        choices=["none", "sqrt_log_combined_favourites", "log_combined_favourites", "sqrt_combined_favourites"],
        default="none",
        help="Optional KMeans sample weighting. Character vectors are still normalized first.",
    )
    parser.add_argument("--shared-role-weight", choices=["none", "inverse_sqrt", "inverse"], default="none")
    return parser.parse_args()


def nnls_pg(atoms: np.ndarray, target: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    count = atoms.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    gram = atoms @ atoms.T
    rhs = atoms @ target
    if count == 1:
        lipschitz = max(float(gram[0, 0]), 1.0e-12)
    else:
        vector = np.ones(count, dtype=np.float64)
        vector /= np.linalg.norm(vector)
        for _ in range(40):
            vector = gram @ vector
            vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
        lipschitz = max(float(vector @ (gram @ vector)), 1.0e-12)

    coefficients = np.zeros(count, dtype=np.float64)
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


def positive_descriptor_fit(
    target: np.ndarray,
    descriptor_atoms: np.ndarray,
    descriptors: list[str],
    *,
    stop_fit: float,
    min_terms: int,
    max_terms: int,
    candidate_indices: np.ndarray | None = None,
) -> dict:
    target = target / max(float(np.linalg.norm(target)), 1.0e-12)
    if candidate_indices is not None:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        descriptor_atoms = descriptor_atoms[candidate_indices]
        descriptors = [descriptors[int(index)] for index in candidate_indices]
    atoms = normalize_rows(descriptor_atoms)
    active: list[int] = []
    selected: set[int] = set()
    residual = target.copy()
    coefficients = np.zeros(0, dtype=np.float64)
    previous_sse = float(residual @ residual)
    steps = []

    for step in range(1, max_terms + 1):
        correlations = atoms @ residual
        if selected:
            correlations[list(selected)] = -np.inf
        descriptor_index = int(np.argmax(correlations))
        if not np.isfinite(correlations[descriptor_index]) or correlations[descriptor_index] <= 0.0:
            steps.append({"step": step, "stop": "no_positive_residual_descriptor"})
            break

        active.append(descriptor_index)
        selected.add(descriptor_index)
        coefficients = nnls_pg(atoms[active], target)
        approximation = coefficients @ atoms[active]
        residual = target - approximation
        sse = float(residual @ residual)
        fit = float(approximation @ target / max(float(np.linalg.norm(approximation)), 1.0e-12))
        gain = (previous_sse - sse) / previous_sse if previous_sse > 0.0 else 0.0
        steps.append(
            {
                "step": step,
                "descriptor": descriptors[descriptor_index],
                "fit_percent": round(fit * 100.0, 6),
                "residual_gain_percent": round(gain * 100.0, 6),
            }
        )
        previous_sse = sse
        if fit >= stop_fit and step >= min_terms:
            break

    if not active:
        return {"fit_percent": 0.0, "descriptors": [], "steps": steps}

    approximation = coefficients @ atoms[active]
    fit = float(approximation @ target / max(float(np.linalg.norm(approximation)), 1.0e-12))
    total = max(float(np.sum(coefficients)), 1.0e-12)
    rows = []
    for rank, coefficient_index in enumerate(np.argsort(coefficients)[::-1], start=1):
        coefficient = float(coefficients[int(coefficient_index)])
        if coefficient <= 1.0e-8:
            continue
        rows.append(
            {
                "rank": rank,
                "descriptor": descriptors[active[int(coefficient_index)]],
                "coefficient": round(coefficient, 10),
                "percent": round(coefficient / total * 100.0, 6),
            }
        )
    return {
        "fit_percent": round(fit * 100.0, 6),
        "terms": len(rows),
        "descriptors": rows,
        "steps": steps,
    }


def character_payload(character: dict, *, distance: float | None = None, similarity: float | None = None) -> dict:
    row = {
        "character_id": character["character_id"],
        "name": character["name"],
        "anime": character["anime"],
        "image": character.get("image") or "",
        "site_url": character.get("site_url") or "",
        "combined_favourites": int(character.get("combined_favourites") or 0),
        "anilist_favourites": int(character.get("anilist_favourites") or character.get("favourites") or 0),
        "bangumi_collects": int(character.get("bangumi_collects") or 0),
        "descriptors": character.get("descriptors") or [],
    }
    if distance is not None:
        row["cosine_distance_to_centroid"] = round(float(distance), 8)
    if similarity is not None:
        row["cosine_similarity_to_centroid"] = round(float(similarity), 8)
    return row


def build_seiyuu_character_rows(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    site_profiles = read_json(args.site_profile_input)["profiles"]
    tag_payload = read_json(args.tags_input)
    profile = find_profile(site_profiles, args.seiyuu)
    safe_tags = load_safe_llm_personality(args.safe_llm_tags)
    bangumi_collects = load_bangumi_collects(args.safe_enrichment, args.bangumi_raw_dir)
    display_payload = read_json(args.character_display_input)
    display_by_id = {int(key): value for key, value in display_payload.get("characters", {}).items()}
    role_display_by_id = load_role_character_display(args.role_edges)

    seiyuu_to_characters: dict[str, list[dict]] = defaultdict(list)
    for source in tag_payload.get("characters", []):
        if is_pop_team_character(source) or len(source.get("seiyuu") or []) > args.max_role_edge_count:
            continue
        tag_rows = descriptor_rows_from_character(source, safe_tags)
        if not tag_rows:
            continue
        character = enrich_character(
            source,
            display_by_id,
            role_display_by_id,
            bangumi_collects.get(int(source["anilist_character_id"]), {}),
        )
        descriptor_weights: dict[str, float] = {}
        descriptor_sources: dict[str, list[dict]] = defaultdict(list)
        for tag_row in tag_rows:
            descriptor = tag_row["tag"]
            descriptor_weights[descriptor] = max(descriptor_weights.get(descriptor, 0.0), 1.0)
            descriptor_sources[descriptor].append(
                {
                    "source": tag_row.get("source"),
                    "source_key": tag_row.get("source_key"),
                    "source_url": tag_row.get("source_url"),
                    "evidence": tag_row.get("evidence"),
                }
            )
        character["descriptor_weights"] = descriptor_weights
        character["descriptor_sources"] = descriptor_sources
        character["descriptors"] = sorted(descriptor_weights)
        for seiyuu in source.get("seiyuu", []):
            for key in name_keys(seiyuu.get("name") or ""):
                seiyuu_to_characters[key].append(character)

    seen = set()
    characters = []
    for key in alias_keys(profile):
        for character in seiyuu_to_characters.get(key, []):
            if character["character_id"] in seen:
                continue
            seen.add(character["character_id"])
            characters.append(character)
    if len(characters) < 2:
        raise RuntimeError(f"Need at least two described characters for {profile['name']}; found {len(characters)}.")
    return profile, characters


def lowdin_coordinates(B: np.ndarray, G: np.ndarray, regularization: float) -> tuple[np.ndarray, np.ndarray, dict]:
    eigenvalues, eigenvectors = np.linalg.eigh((G + G.T) * 0.5)
    max_eval = max(float(eigenvalues.max(initial=0.0)), 1.0)
    reg = max_eval * regularization
    factors = np.divide(
        eigenvalues,
        np.sqrt(np.maximum(eigenvalues + reg, reg)),
        out=np.zeros_like(eigenvalues),
        where=eigenvalues > 0.0,
    )
    descriptor_coordinates = eigenvectors @ np.diag(factors)
    Z = B @ descriptor_coordinates
    positive = eigenvalues[eigenvalues > 0.0]
    min_positive = float(positive.min()) if len(positive) else 0.0
    return Z, descriptor_coordinates, {
        "rank": int(len(positive)),
        "regularization_absolute": reg,
        "min_positive_eigenvalue": min_positive,
        "max_eigenvalue": float(max_eval),
    }


def descriptor_representative(descriptors: list[str]) -> str:
    def key(value: str) -> tuple[int, int, str]:
        parts = value.replace("-", " ").split()
        return (len(parts), len(value), value)

    return sorted(descriptors, key=key)[0]


def descriptor_tokens(value: str) -> list[str]:
    normalized = value.lower().replace("-", " ").replace("_", " ")
    return [token for token in normalized.split() if token]


def contains_token_subsequence(longer: str, shorter: str) -> bool:
    longer_tokens = descriptor_tokens(longer)
    shorter_tokens = descriptor_tokens(shorter)
    if not shorter_tokens or len(shorter_tokens) >= len(longer_tokens):
        return False
    width = len(shorter_tokens)
    return any(longer_tokens[start : start + width] == shorter_tokens for start in range(len(longer_tokens) - width + 1))


def canonicalize_descriptors(
    descriptors: list[str],
    embeddings: np.ndarray,
    similarity_threshold: float,
    contained_distance_threshold: float,
) -> tuple[list[str], np.ndarray, dict[str, str], list[dict]]:
    if (similarity_threshold > 1.0 and contained_distance_threshold < 0.0) or len(descriptors) <= 1:
        return descriptors, embeddings, {descriptor: descriptor for descriptor in descriptors}, []

    descriptor_to_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    parent = list(range(len(descriptors)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    if similarity_threshold <= 1.0:
        for left in range(len(descriptors)):
            for right in range(left + 1, len(descriptors)):
                if float(similarity[left, right]) >= similarity_threshold:
                    union(left, right)

    groups_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(descriptors)):
        groups_by_root[find(index)].append(index)

    raw_to_canonical = {}
    for indices in groups_by_root.values():
        raw_group = [descriptors[index] for index in indices]
        representative = descriptor_representative(raw_group)
        for raw_descriptor in raw_group:
            raw_to_canonical[raw_descriptor] = representative

    if contained_distance_threshold >= 0.0:
        for left in range(len(descriptors)):
            candidates = []
            for right in range(len(descriptors)):
                if left == right:
                    continue
                if contains_token_subsequence(descriptors[left], descriptors[right]):
                    distance = 1.0 - float(similarity[left, right])
                    if distance <= contained_distance_threshold:
                        candidates.append((distance, len(descriptor_tokens(descriptors[right])), len(descriptors[right]), descriptors[right]))
            if candidates:
                _, _, _, best_shorter = min(candidates)
                raw_to_canonical[descriptors[left]] = raw_to_canonical.get(best_shorter, best_shorter)
        # Resolve chains like "very classic tsundere" -> "classic tsundere" -> "tsundere".
        for raw_descriptor in list(raw_to_canonical):
            seen = set()
            canonical = raw_to_canonical[raw_descriptor]
            while canonical in raw_to_canonical and raw_to_canonical[canonical] != canonical and canonical not in seen:
                seen.add(canonical)
                canonical = raw_to_canonical[canonical]
            raw_to_canonical[raw_descriptor] = canonical

    canonical_to_members: dict[str, list[str]] = defaultdict(list)
    for raw_descriptor, canonical in raw_to_canonical.items():
        canonical_to_members[canonical].append(raw_descriptor)

    canonical_descriptors = sorted(canonical_to_members)
    canonical_embeddings = []
    merge_groups = []
    for canonical in canonical_descriptors:
        member_indices = [descriptor_to_index[member] for member in canonical_to_members[canonical]]
        if canonical in descriptor_to_index:
            canonical_embedding = embeddings[descriptor_to_index[canonical]]
        else:
            canonical_embedding = normalize_rows(embeddings[member_indices].mean(axis=0, keepdims=True))[0]
        canonical_embeddings.append(canonical_embedding)
        raw_group = sorted(canonical_to_members[canonical])
        if len(raw_group) > 1:
            merge_groups.append(
                {
                    "canonical": canonical,
                    "members": raw_group,
                    "size": len(raw_group),
                }
            )

    return (
        canonical_descriptors,
        np.asarray(canonical_embeddings, dtype=np.float64),
        raw_to_canonical,
        sorted(merge_groups, key=lambda row: (-row["size"], row["canonical"])),
    )


def apply_descriptor_canonicalization(characters: list[dict], raw_to_canonical: dict[str, str]) -> None:
    for character in characters:
        canonical_weights: dict[str, float] = {}
        canonical_sources: dict[str, list[dict]] = defaultdict(list)
        for raw_descriptor, weight in character.get("descriptor_weights", {}).items():
            canonical = raw_to_canonical.get(raw_descriptor, raw_descriptor)
            canonical_weights[canonical] = max(canonical_weights.get(canonical, 0.0), float(weight))
            canonical_sources[canonical].extend(character.get("descriptor_sources", {}).get(raw_descriptor, []))
        character["descriptor_weights"] = canonical_weights
        character["descriptor_sources"] = canonical_sources
        character["descriptors"] = sorted(canonical_weights)


def load_global_raw_to_canonical(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return {
        str(raw): str(canonical)
        for raw, canonical in (payload.get("raw_to_canonical") or {}).items()
        if raw and canonical
    }


def scan_kmeans(Z_unit: np.ndarray, args: argparse.Namespace, sample_weight: np.ndarray | None) -> tuple[np.ndarray, KMeans, list[dict]]:
    scans = []
    best_model = None
    best_output_labels = None
    fallback_model = None
    fallback_labels = None
    fallback_score = (np.inf, 0)
    upper = max(args.min_k, min(args.max_k, len(Z_unit) - 1))
    for k in range(args.min_k, upper + 1):
        model = KMeans(n_clusters=k, random_state=args.random_state, n_init=args.n_init)
        labels = model.fit_predict(Z_unit, sample_weight=sample_weight)
        counts = np.bincount(labels, minlength=k)
        if len(set(labels)) > 1 and min(counts) > 0:
            silhouette = float(silhouette_score(Z_unit, labels, metric="cosine"))
        else:
            silhouette = -1.0
        centroids = normalize_rows(model.cluster_centers_)
        assigned_similarity = np.asarray([float(Z_unit[i] @ centroids[labels[i]]) for i in range(len(Z_unit))])
        assigned_distance = np.clip(1.0 - assigned_similarity, 0.0, 2.0)
        mean_distance = float(assigned_distance.mean())
        max_distance = float(assigned_distance.max())
        radius_ratio = max_distance / max(mean_distance, 1.0e-12)
        valid_by_radius = bool(radius_ratio <= args.radius_ratio_threshold and int(counts.min()) >= args.min_cluster_size)
        scans.append(
            {
                "k": k,
                "silhouette_cosine": round(silhouette, 8),
                "inertia": round(float(model.inertia_), 8),
                "mean_cosine_distance_to_centroid": round(mean_distance, 8),
                "max_cosine_distance_to_centroid": round(max_distance, 8),
                "radius_ratio": round(radius_ratio, 8),
                "valid_by_radius_rule": valid_by_radius,
                "cluster_sizes": [int(value) for value in counts],
            }
        )
        if valid_by_radius and best_model is None:
            best_model = model
            best_output_labels = labels
        fallback_candidate = (radius_ratio, k)
        if fallback_model is None or fallback_candidate < fallback_score:
            fallback_model = model
            fallback_labels = labels
            fallback_score = fallback_candidate
    if best_model is None:
        best_model = fallback_model
        best_output_labels = fallback_labels
    assert best_model is not None
    assert best_output_labels is not None
    return best_output_labels, best_model, scans


def centroid_for_rows(rows: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    if sample_weight is None:
        centroid = rows.mean(axis=0)
    else:
        centroid = np.average(rows, axis=0, weights=sample_weight)
    norm = max(float(np.linalg.norm(centroid)), 1.0e-12)
    return centroid / norm


def mean_distance_to_centroid(rows: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, float, np.ndarray]:
    centroid = centroid_for_rows(rows, sample_weight)
    distances = np.clip(1.0 - rows @ centroid, 0.0, 2.0)
    if sample_weight is None:
        mean_distance = float(distances.mean())
    else:
        mean_distance = float(np.average(distances, weights=sample_weight))
    return mean_distance, float(distances.max()), centroid


def recursive_split_clusters(
    Z_unit: np.ndarray,
    args: argparse.Namespace,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    leaves: list[np.ndarray] = []
    trimmed: list[np.ndarray] = []
    split_log: list[dict] = []

    def split_node(indices: np.ndarray, depth: int) -> None:
        local_weights = sample_weight[indices] if sample_weight is not None else None
        parent_mean, parent_max, _ = mean_distance_to_centroid(Z_unit[indices], local_weights)
        if (
            depth >= args.max_recursive_depth
            or len(indices) < args.min_cluster_size * 2
            or len(leaves) + 1 >= args.max_k
        ):
            leaves.append(indices)
            return

        model = KMeans(n_clusters=2, random_state=args.random_state + depth, n_init=args.n_init)
        local_labels = model.fit_predict(Z_unit[indices], sample_weight=local_weights)
        child_positions = [np.flatnonzero(local_labels == child_id) for child_id in range(2)]
        child_sizes = [int(len(position)) for position in child_positions]
        child_indices = [indices[position] for position in child_positions]
        child_means = []
        child_centroids = []
        child_maxes = []
        for child_index in child_indices:
            child_weights = sample_weight[child_index] if sample_weight is not None else None
            mean_distance, max_distance, centroid = mean_distance_to_centroid(Z_unit[child_index], child_weights)
            child_means.append(mean_distance)
            child_maxes.append(max_distance)
            child_centroids.append(centroid)

        if sample_weight is None:
            child_mean = float(sum(len(child) * mean for child, mean in zip(child_indices, child_means)) / len(indices))
        else:
            total_weight = float(np.sum(sample_weight[indices]))
            child_mean = float(
                sum(float(np.sum(sample_weight[child])) * mean for child, mean in zip(child_indices, child_means))
                / max(total_weight, 1.0e-12)
            )
        split_gain = (parent_mean - child_mean) / max(parent_mean, 1.0e-12)
        child_centroid_distance = float(np.clip(1.0 - child_centroids[0] @ child_centroids[1], 0.0, 2.0))
        accepted = (
            min(child_sizes) >= args.min_cluster_size
            and split_gain >= args.min_split_gain
            and child_centroid_distance >= args.min_split_distance
        )
        small_child_id = int(np.argmin(child_sizes))
        large_child_id = int(np.argmax(child_sizes))
        can_trim = (
            not args.no_trim_outliers
            and not accepted
            and child_sizes[small_child_id] < args.min_cluster_size
            and child_sizes[large_child_id] >= args.min_cluster_size * 2
            and split_gain >= args.min_split_gain
            and child_centroid_distance >= args.min_split_distance
        )
        split_log.append(
            {
                "depth": depth,
                "size": int(len(indices)),
                "parent_mean_cosine_distance": round(parent_mean, 8),
                "parent_max_cosine_distance": round(parent_max, 8),
                "child_sizes": child_sizes,
                "child_mean_cosine_distances": [round(value, 8) for value in child_means],
                "child_max_cosine_distances": [round(value, 8) for value in child_maxes],
                "weighted_child_mean_cosine_distance": round(child_mean, 8),
                "split_gain": round(split_gain, 8),
                "child_centroid_cosine_distance": round(child_centroid_distance, 8),
                "accepted": accepted,
                "trimmed_outlier_child": bool(can_trim),
            }
        )
        if can_trim:
            trimmed.append(child_indices[small_child_id])
            split_node(child_indices[large_child_id], depth + 1)
            return
        if not accepted:
            leaves.append(indices)
            return

        order = np.argsort([-len(child) for child in child_indices])
        for child_offset in order:
            split_node(child_indices[int(child_offset)], depth + 1)

    split_node(np.arange(len(Z_unit)), 0)
    if trimmed and leaves:
        leaf_centroids = [
            centroid_for_rows(Z_unit[indices], sample_weight[indices] if sample_weight is not None else None)
            for indices in leaves
        ]
        for trim_group in trimmed:
            for index in trim_group:
                similarities = [float(Z_unit[int(index)] @ centroid) for centroid in leaf_centroids]
                leaf_id = int(np.argmax(similarities))
                leaves[leaf_id] = np.concatenate([leaves[leaf_id], np.asarray([int(index)], dtype=np.int64)])
                leaf_centroids[leaf_id] = centroid_for_rows(
                    Z_unit[leaves[leaf_id]],
                    sample_weight[leaves[leaf_id]] if sample_weight is not None else None,
                )
    leaves.sort(key=lambda row: (-len(row), int(row.min()) if len(row) else 0))
    labels = np.empty(len(Z_unit), dtype=np.int64)
    for cluster_id, indices in enumerate(leaves):
        labels[indices] = cluster_id
    centers = np.vstack([centroid_for_rows(Z_unit[indices], sample_weight[indices] if sample_weight is not None else None) for indices in leaves])
    model = KMeans(n_clusters=len(leaves), random_state=args.random_state, n_init=1)
    model.cluster_centers_ = centers
    model.labels_ = labels
    model.inertia_ = float(
        sum(
            np.sum((Z_unit[indices] - centers[cluster_id]) ** 2)
            for cluster_id, indices in enumerate(leaves)
        )
    )
    return labels, {"model": model, "split_log": split_log, "trimmed_count": int(sum(len(row) for row in trimmed))}


def descriptor_rows_for_cluster(
    B: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    descriptors: list[str],
    descriptor_embeddings: np.ndarray,
    centroid: np.ndarray,
    descriptor_atoms: np.ndarray,
    limit: int,
) -> dict:
    in_mask = labels == cluster_id
    out_mask = labels != cluster_id
    in_mean = B[in_mask].mean(axis=0)
    out_mean = B[out_mask].mean(axis=0) if np.any(out_mask) else np.zeros(B.shape[1])
    enrichment = in_mean - out_mean
    candidate_indices = np.flatnonzero((enrichment > 0.0) & (in_mean > 0.0))
    enriched_order = np.argsort(enrichment)[::-1]
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1.0e-12)
    atom_similarity = normalize_rows(descriptor_atoms) @ centroid
    nearest_order = np.argsort(atom_similarity)[::-1]
    return {
        "fit_candidate_indices": candidate_indices,
        "enriched_descriptors": [
            {
                "descriptor": descriptors[int(index)],
                "score": round(float(enrichment[int(index)]), 8),
                "cluster_mean": round(float(in_mean[int(index)]), 8),
                "outside_mean": round(float(out_mean[int(index)]), 8),
            }
            for index in enriched_order[:limit]
            if float(enrichment[int(index)]) > 0.0
        ],
        "nearest_descriptors": [
            {
                "descriptor": descriptors[int(index)],
                "similarity": round(float(atom_similarity[int(index)]), 8),
            }
            for index in nearest_order[:limit]
        ],
    }


def cluster_payloads(
    Z_unit: np.ndarray,
    B: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
    characters: list[dict],
    descriptors: list[str],
    descriptor_embeddings: np.ndarray,
    descriptor_atoms: np.ndarray,
    top_descriptors: int,
    top_characters: int,
    descriptor_fit_target: float,
    descriptor_fit_min_terms: int,
    descriptor_fit_max_terms: int,
) -> list[dict]:
    clusters = []
    centroids = normalize_rows(model.cluster_centers_)
    similarity_matrix = Z_unit @ Z_unit.T
    for cluster_id in range(model.n_clusters):
        positions = np.flatnonzero(labels == cluster_id)
        centroid = centroids[cluster_id]
        similarities = Z_unit[positions] @ centroid
        order = positions[np.argsort(similarities)[::-1]]
        internal = similarity_matrix[np.ix_(positions, positions)]
        outside = np.flatnonzero(labels != cluster_id)
        external = similarity_matrix[np.ix_(positions, outside)] if len(outside) else np.zeros((len(positions), 0))
        descriptor_payload = descriptor_rows_for_cluster(
            B,
            labels,
            cluster_id,
            descriptors,
            descriptor_embeddings,
            centroid,
            descriptor_atoms,
            top_descriptors,
        )
        positive_fit = positive_descriptor_fit(
            centroid,
            descriptor_atoms,
            descriptors,
            stop_fit=descriptor_fit_target,
            min_terms=descriptor_fit_min_terms,
            max_terms=descriptor_fit_max_terms,
            candidate_indices=descriptor_payload.pop("fit_candidate_indices"),
        )
        clusters.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(positions)),
                "mean_internal_similarity": round(float(internal[np.triu_indices(len(positions), 1)].mean()), 8)
                if len(positions) > 1
                else 1.0,
                "mean_external_similarity": round(float(external.mean()), 8) if external.size else 0.0,
                "positive_descriptor_fit": positive_fit,
                "dominant_descriptors": positive_fit["descriptors"],
                **descriptor_payload,
                "characters": [
                    character_payload(
                        characters[int(index)],
                        distance=1.0 - float(Z_unit[int(index)] @ centroid),
                        similarity=float(Z_unit[int(index)] @ centroid),
                    )
                    for index in order[:top_characters]
                ],
            }
        )
    clusters.sort(key=lambda row: (-row["size"], row["cluster"]))
    return clusters


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile, characters = build_seiyuu_character_rows(args)
    raw_descriptors = sorted({descriptor for character in characters for descriptor in character["descriptors"]})
    global_raw_to_canonical = load_global_raw_to_canonical(args.global_canonicalization_input)
    if global_raw_to_canonical:
        raw_to_canonical = {
            descriptor: global_raw_to_canonical.get(descriptor, descriptor)
            for descriptor in raw_descriptors
        }
        canonical_groups = []
        descriptors = sorted(set(raw_to_canonical.values()))
        apply_descriptor_canonicalization(characters, raw_to_canonical)
        E = load_or_create_embeddings(descriptors, args.output_dir, args.embedding_model)
    else:
        raw_E = load_or_create_embeddings(raw_descriptors, args.output_dir, args.embedding_model)
        descriptors, E, raw_to_canonical, canonical_groups = canonicalize_descriptors(
            raw_descriptors,
            raw_E,
            args.canonicalize_similarity_threshold,
            args.canonicalize_contained_distance_threshold,
        )
        apply_descriptor_canonicalization(characters, raw_to_canonical)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    G = np.clip(E @ E.T, -1.0, 1.0)
    B = np.zeros((len(characters), len(descriptors)), dtype=np.float64)
    for row_index, character in enumerate(characters):
        for descriptor, weight in character["descriptor_weights"].items():
            B[row_index, descriptor_index[descriptor]] = float(weight)
    Z, descriptor_coordinates, lowdin = lowdin_coordinates(B, G, args.regularization)
    Z_unit = normalize_rows(Z)
    semantic_overlap = np.clip(Z_unit @ Z_unit.T, -1.0, 1.0)
    cosine_distance = np.clip(1.0 - semantic_overlap, 0.0, 2.0)
    sample_weight = None
    if args.row_weight != "none" or args.shared_role_weight != "none":
        sample_weight = np.asarray(
            [
                character_row_weight(character, args.row_weight) * shared_role_weight(character, args.shared_role_weight)
                for character in characters
            ],
            dtype=np.float64,
        )
    split_log = []
    trimmed_count = 0
    if args.cluster_method == "recursive":
        labels, recursive = recursive_split_clusters(Z_unit, args, sample_weight)
        model = recursive["model"]
        split_log = recursive["split_log"]
        trimmed_count = recursive["trimmed_count"]
        _, _, scans = scan_kmeans(Z_unit, args, sample_weight)
    else:
        labels, model, scans = scan_kmeans(Z_unit, args, sample_weight)

    clusters = cluster_payloads(
        Z_unit,
        B,
        labels,
        model,
        characters,
        descriptors,
        E,
        descriptor_coordinates,
        args.top_descriptors,
        args.top_characters,
        args.descriptor_fit_target,
        args.descriptor_fit_min_terms,
        args.descriptor_fit_max_terms,
    )
    output = {
        "generated_at": utc_now(),
        "source": "seiyuu_character_semantic_clusters.py",
        "parameters": {
            "seiyuu_query": args.seiyuu,
            "profile_name": profile["name"],
            "matrix": "Z = B @ G @ X, with G=E@E.T and X=U diag(1/sqrt(lambda + regularization)) from G's eigendecomposition; rows normalized to unit overlap before KMeans.",
            "distance": "For normalized character rows, semantic overlap is cosine similarity Z_unit @ Z_unit.T; cosine distance is 1 - overlap.",
            "embedding_model": args.embedding_model,
            "regularization": args.regularization,
            "canonicalize_similarity_threshold": args.canonicalize_similarity_threshold,
            "canonicalize_contained_distance_threshold": args.canonicalize_contained_distance_threshold,
            "global_canonicalization_input": str(args.global_canonicalization_input),
            "used_global_canonicalization": bool(global_raw_to_canonical),
            "k_selection": f"smallest K with radius_ratio <= {args.radius_ratio_threshold} and min cluster size >= {args.min_cluster_size}; fallback is lowest radius_ratio.",
            "cluster_descriptor_fit": f"greedy nonnegative fit of each normalized cluster centroid using original local descriptor atoms; stop at cosine fit >= {args.descriptor_fit_target} or {args.descriptor_fit_max_terms} terms.",
            "cluster_method": args.cluster_method,
            "recursive_split_rule": f"accept a 2-way split only if both children have >= {args.min_cluster_size} characters, mean cosine distance drops by >= {args.min_split_gain}, and child centroids are at least {args.min_split_distance} cosine-distance apart.",
            "trim_outliers_before_core_splitting": not args.no_trim_outliers,
            "row_weight": args.row_weight,
            "shared_role_weight": args.shared_role_weight,
            "k_scan": [args.min_k, args.max_k],
            "random_state": args.random_state,
        },
        "counts": {
            "characters": len(characters),
            "descriptors": len(descriptors),
            "raw_descriptors": len(raw_descriptors),
            "canonical_descriptor_groups": len(canonical_groups),
            **lowdin,
        },
        "canonical_descriptor_groups": canonical_groups,
        "kmeans_scan": scans,
        "recursive_split_log": split_log,
        "recursive_trimmed_count": trimmed_count,
        "chosen_k": int(model.n_clusters),
        "semantic_overlap_matrix": [
            [round(float(value), 8) for value in row]
            for row in semantic_overlap
        ],
        "cosine_distance_matrix": [
            [round(float(value), 8) for value in row]
            for row in cosine_distance
        ],
        "characters": [character_payload(character) for character in characters],
        "clusters": clusters,
    }
    output_path = args.output_dir / f"{slug(profile['name'])}_character_semantic_clusters.json"
    write_json(output_path, output)
    print(f"wrote {output_path}")
    print(json.dumps({"counts": output["counts"], "chosen_k": output["chosen_k"], "scan": scans}, indent=2))
    for cluster in clusters:
        print()
        print(
            "cluster",
            cluster["cluster"],
            "size",
            cluster["size"],
            "chars",
            [row["name"] for row in cluster["characters"][:8]],
        )
        print("  enriched", [row["descriptor"] for row in cluster["enriched_descriptors"][:8]])
        print(
            "  fitted",
            round(cluster["positive_descriptor_fit"]["fit_percent"], 2),
            [row["descriptor"] for row in cluster["positive_descriptor_fit"]["descriptors"][:8]],
        )
        print("  nearest", [row["descriptor"] for row in cluster["nearest_descriptors"][:8]])


if __name__ == "__main__":
    main()
