#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from importlib.metadata import version, PackageNotFoundError

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine semantic character lanes from descriptor-embedding distances.")
    parser.add_argument("--seiyuu", default="Ayana Taketatsu")
    parser.add_argument("--role-cache", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--semantic-model-dir", type=Path, default=Path("models/semantic_wordnet_descriptors"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/semantic_character_lanes"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-features-per-character", type=int, default=18)
    parser.add_argument("--min-feature-weight", type=float, default=1e-6)
    parser.add_argument("--distance-channel", choices=("style", "all"), default="style")
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--top-cluster-features", type=int, default=14)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def load_or_create_embeddings(features: list[str], output_dir: Path, model_name: str) -> np.ndarray:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    cache_path = output_dir / f"feature_embeddings_{safe_name}.npz"
    vocab_path = output_dir / f"feature_embeddings_{safe_name}.json"
    if cache_path.exists() and vocab_path.exists():
        cached_vocab = read_json(vocab_path)["features"]
        if cached_vocab == features:
            return np.load(cache_path)["embeddings"].astype(np.float32)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        features,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    write_json(vocab_path, {"model": model_name, "features": features})
    return embeddings


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def feature_channel_metadata(semantic_model_dir: Path, feature_names: list[str]) -> tuple[np.ndarray, list[dict]]:
    details_path = semantic_model_dir / "semantic_features_by_character.json"
    feature_stats: dict[str, dict] = defaultdict(lambda: {"kinds": Counter(), "lexnames": Counter()})
    if details_path.exists():
        for row in read_json(details_path):
            for feature in row.get("features", []):
                stats = feature_stats[feature["feature"]]
                stats["kinds"][feature.get("kind", "")] += 1
                stats["lexnames"][feature.get("lexname", "")] += 1

    style_mask = np.zeros(len(feature_names), dtype=bool)
    metadata = []
    for index, feature in enumerate(feature_names):
        stats = feature_stats.get(feature, {"kinds": Counter(), "lexnames": Counter()})
        kinds: Counter = stats["kinds"]
        lexnames: Counter = stats["lexnames"]
        style_signal = sum(count for lexname, count in lexnames.items() if str(lexname).startswith("adj."))
        style_signal += kinds.get("attribute_modifier", 0)
        style_signal += kinds.get("morphological_like", 0)
        noun_signal = sum(
            count
            for lexname, count in lexnames.items()
            if str(lexname).startswith("noun.") and lexname not in {"noun.attribute"}
        )
        unknown_modifier_signal = kinds.get("surface_modifier", 0) + lexnames.get("adj_or_unknown", 0)
        is_style = style_signal > 0 and style_signal > noun_signal
        style_mask[index] = bool(is_style)
        metadata.append(
            {
                "feature": feature,
                "style_signal": int(style_signal),
                "noun_signal": int(noun_signal),
                "unknown_modifier_signal": int(unknown_modifier_signal),
                "is_style": bool(is_style),
                "kinds": dict(kinds),
                "lexnames": dict(lexnames),
            }
        )
    return style_mask, metadata


def role_payload(role: dict, weight: float | None = None) -> dict:
    character = role["character"]
    anime = role.get("anime") or []
    first_anime = anime[0] if anime else {}
    payload = {
        "character_id": int(character["character_id"]),
        "name": character.get("name") or "",
        "image": character.get("image") or "",
        "site_url": character.get("site_url") or "",
        "role": role.get("character_role") or "",
        "anime_title": first_anime.get("title") or character.get("first_anime") or "",
        "anime_url": first_anime.get("mal_url") or first_anime.get("site_url") or "",
    }
    if weight is not None:
        payload["score"] = round(float(weight), 6)
    return payload


def selected_feature_indices(row, limit: int, min_weight: float) -> tuple[np.ndarray, np.ndarray]:
    if row.nnz == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
    indices = row.indices
    weights = row.data.astype(np.float64)
    keep = weights > min_weight
    indices = indices[keep]
    weights = weights[keep]
    if len(indices) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
    order = np.argsort(weights)[::-1][:limit]
    indices = indices[order]
    weights = weights[order]
    weights = weights / max(float(weights.sum()), 1e-12)
    return indices.astype(np.int64), weights.astype(np.float32)


def symmetric_soft_distance(
    left_indices: np.ndarray,
    left_weights: np.ndarray,
    right_indices: np.ndarray,
    right_weights: np.ndarray,
    embeddings: np.ndarray,
) -> float:
    if len(left_indices) == 0 and len(right_indices) == 0:
        return 0.0
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 1.0
    sim = embeddings[left_indices] @ embeddings[right_indices].T
    cost = np.clip(1.0 - sim, 0.0, 2.0)
    left_to_right = float(np.sum(left_weights * cost.min(axis=1)))
    right_to_left = float(np.sum(right_weights * cost.min(axis=0)))
    return 0.5 * (left_to_right + right_to_left)


def distance_matrix(feature_sets: list[tuple[np.ndarray, np.ndarray]], embeddings: np.ndarray) -> np.ndarray:
    n = len(feature_sets)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            distance = symmetric_soft_distance(*feature_sets[i], *feature_sets[j], embeddings)
            distances[i, j] = distances[j, i] = distance
    return distances


def spectral_coordinates(distances: np.ndarray, max_clusters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonzero = distances[distances > 0]
    sigma = float(np.median(nonzero)) if nonzero.size else 1.0
    sigma = max(sigma, 1e-6)
    affinity = np.exp(-distances / sigma).astype(np.float64)
    np.fill_diagonal(affinity, 0.0)
    degrees = affinity.sum(axis=1)
    inv_sqrt = np.divide(1.0, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees > 0)
    laplacian = np.eye(len(distances)) - (inv_sqrt[:, None] * affinity * inv_sqrt[None, :])
    eigenvalues, eigenvectors = eigh(laplacian)
    dims = max(2, min(max_clusters, len(distances) - 1))
    coords = eigenvectors[:, 1 : dims + 1]
    return eigenvalues, coords, affinity


def deterministic_clusters(coords: np.ndarray, max_clusters: int) -> np.ndarray:
    if len(coords) <= 2:
        return np.ones(len(coords), dtype=np.int64)
    k = max(2, min(max_clusters, len(coords)))
    condensed = pdist(coords[:, : max(1, min(k, coords.shape[1]))], metric="euclidean")
    tree = linkage(condensed, method="ward")
    return fcluster(tree, t=k, criterion="maxclust").astype(np.int64)


def cluster_payload(
    labels: np.ndarray,
    roles: list[dict],
    semantic_matrix,
    feature_names: list[str],
    style_mask: np.ndarray,
    min_cluster_size: int,
    top_features: int,
) -> list[dict]:
    clusters = []
    for label in sorted(set(int(x) for x in labels)):
        positions = np.where(labels == label)[0]
        if len(positions) < min_cluster_size:
            continue
        pooled = np.asarray(semantic_matrix[positions].sum(axis=0)).ravel()
        feature_order = np.argsort(pooled)[::-1]
        features = [
            {"feature": feature_names[int(index)], "weight": round(float(pooled[int(index)]), 6)}
            for index in feature_order[:top_features]
            if float(pooled[int(index)]) > 0
        ]
        style_features = [
            {"feature": feature_names[int(index)], "weight": round(float(pooled[int(index)]), 6)}
            for index in feature_order
            if float(pooled[int(index)]) > 0 and bool(style_mask[int(index)])
        ][:top_features]
        facet_features = [
            {"feature": feature_names[int(index)], "weight": round(float(pooled[int(index)]), 6)}
            for index in feature_order
            if float(pooled[int(index)]) > 0 and not bool(style_mask[int(index)])
        ][:top_features]
        clusters.append(
            {
                "cluster": int(label),
                "size": int(len(positions)),
                "characters": [role_payload(roles[int(position)]) for position in positions],
                "features": features,
                "style_features": style_features,
                "facet_features": facet_features,
            }
        )
    clusters.sort(key=lambda cluster: (-cluster["size"], cluster["cluster"]))
    return clusters


def nearest_neighbors(distances: np.ndarray, roles: list[dict], target_names: list[str], limit: int = 8) -> dict:
    output = {}
    name_to_position = {role["character"].get("name"): i for i, role in enumerate(roles)}
    for name in target_names:
        if name not in name_to_position:
            continue
        position = name_to_position[name]
        order = np.argsort(distances[position])
        output[name] = [
            {**role_payload(roles[int(i)]), "distance": round(float(distances[position, int(i)]), 6)}
            for i in order[:limit]
        ]
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_json(args.semantic_model_dir / "character_rows.json")
    feature_names = [row["feature"] for row in read_json(args.semantic_model_dir / "semantic_vocabulary.json")]
    style_mask, feature_channel_info = feature_channel_metadata(args.semantic_model_dir, feature_names)
    semantic_matrix = sparse.load_npz(args.semantic_model_dir / "character_semantic_tfidf.npz").tocsr()
    role_cache = read_json(args.role_cache)
    char_to_row = {int(row["character_id"]): i for i, row in enumerate(rows)}
    seiyuu_norm = norm_name(args.seiyuu)

    roles = []
    seen = set()
    for role in role_cache["roles"]:
        if norm_name(role["seiyuu"].get("name") or "") != seiyuu_norm:
            continue
        character_id = int(role["character"]["character_id"])
        if character_id not in char_to_row or character_id in seen:
            continue
        seen.add(character_id)
        roles.append(role)
    if not roles:
        raise RuntimeError(f"No in-scope roles found for {args.seiyuu!r}")

    row_indices = np.asarray([char_to_row[int(role["character"]["character_id"])] for role in roles], dtype=np.int64)
    local_matrix = semantic_matrix[row_indices].tocsr()
    embeddings = load_or_create_embeddings(feature_names, args.output_dir, args.embedding_model)
    if args.distance_channel == "style":
        distance_feature_indices = np.flatnonzero(style_mask)
        distance_matrix_source = local_matrix[:, distance_feature_indices].tocsr()
        distance_embeddings = embeddings[distance_feature_indices]
    else:
        distance_feature_indices = np.arange(len(feature_names))
        distance_matrix_source = local_matrix
        distance_embeddings = embeddings
    feature_sets = [
        selected_feature_indices(distance_matrix_source.getrow(index), args.top_features_per_character, args.min_feature_weight)
        for index in range(distance_matrix_source.shape[0])
    ]
    distances = distance_matrix(feature_sets, distance_embeddings)
    eigenvalues, coords, affinity = spectral_coordinates(distances, args.max_clusters)
    labels = deterministic_clusters(coords, args.max_clusters)
    clusters = cluster_payload(labels, roles, local_matrix, feature_names, style_mask, args.min_cluster_size, args.top_cluster_features)
    output = {
        "seiyuu": args.seiyuu,
        "parameters": {
            "embedding_model": args.embedding_model,
            "embedding_cache": {
                "features_json": f"feature_embeddings_{re.sub(r'[^a-zA-Z0-9_.-]+', '_', args.embedding_model)}.json",
                "embeddings_npz": f"feature_embeddings_{re.sub(r'[^a-zA-Z0-9_.-]+', '_', args.embedding_model)}.npz",
                "normalized_embeddings": True,
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": package_version("numpy"),
                "scipy": package_version("scipy"),
                "scikit_learn": package_version("scikit-learn"),
                "sentence_transformers": package_version("sentence-transformers"),
                "torch": package_version("torch"),
            },
            "distance": "symmetric weighted soft nearest-neighbor cosine distance over descriptor embeddings",
            "distance_channel": args.distance_channel,
            "distance_feature_count": int(len(distance_feature_indices)),
            "distance_feature_policy": "style channel uses dominantly adjectival / modifier semantic descriptors; noun-like role/facet descriptors are excluded from distance but retained for cluster explanation",
            "clustering": "normalized graph Laplacian eigenvectors + deterministic Ward clustering",
            "top_features_per_character": args.top_features_per_character,
            "max_clusters": args.max_clusters,
            "min_cluster_size": args.min_cluster_size,
        },
        "counts": {
            "characters": len(roles),
            "features": len(feature_names),
            "clusters": len(clusters),
        },
        "eigenvalues": [round(float(value), 8) for value in eigenvalues[: min(16, len(eigenvalues))]],
        "feature_channels": {
            "style_feature_count": int(style_mask.sum()),
            "facet_feature_count": int((~style_mask).sum()),
            "examples": {
                "style": [feature_names[int(index)] for index in np.flatnonzero(style_mask)[:30]],
                "facet": [feature_names[int(index)] for index in np.flatnonzero(~style_mask)[:30]],
            },
        },
        "nearest_neighbors": nearest_neighbors(distances, roles, ["Nino Nakano", "Kirino Kousaka"], limit=10),
        "clusters": clusters,
    }
    slug = re.sub(r"[^a-z0-9]+", "_", norm_name(args.seiyuu)).strip("_")
    write_json(args.output_dir / f"{slug}_semantic_lanes.json", output)
    np.savez_compressed(args.output_dir / f"{slug}_distance_matrix.npz", distances=distances, affinity=affinity, labels=labels)
    print(f"wrote {args.output_dir / f'{slug}_semantic_lanes.json'}")
    print(f"characters={len(roles)} clusters={len(clusters)}")


if __name__ == "__main__":
    main()
