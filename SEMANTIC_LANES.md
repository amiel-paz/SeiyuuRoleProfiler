# Semantic Character Lanes

This is an experimental replacement for exact n-gram overlap and NMF topic lanes.
The goal is to cluster a seiyuu's characters by semantic similarity between their
descriptor sets, then explain each cluster by pooling the original descriptors.

## Embedding Model

The current prototype uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It is a small frozen sentence-transformer model. Descriptor embeddings are
normalized, so descriptor-to-descriptor distance is cosine distance:

```text
distance(a, b) = 1 - cosine(embedding(a), embedding(b))
```

## Reproducible Pipeline

First build the deterministic semantic descriptor matrix:

```bash
.venv/bin/python scripts/cache_semantic_descriptor_matrix.py
```

This writes:

```text
models/semantic_wordnet_descriptors/character_semantic_tfidf.npz
models/semantic_wordnet_descriptors/semantic_vocabulary.json
models/semantic_wordnet_descriptors/semantic_features_by_character.json
models/semantic_wordnet_descriptors/semantic_matrix_metadata.json
```

Then mine semantic lanes for one seiyuu:

```bash
.venv/bin/python scripts/semantic_character_lanes.py --seiyuu "Ayana Taketatsu"
```

This writes:

```text
models/semantic_character_lanes/feature_embeddings_sentence-transformers_all-MiniLM-L6-v2.json
models/semantic_character_lanes/feature_embeddings_sentence-transformers_all-MiniLM-L6-v2.npz
models/semantic_character_lanes/ayana_taketatsu_distance_matrix.npz
models/semantic_character_lanes/ayana_taketatsu_semantic_lanes.json
```

The feature embedding cache is reused when the feature vocabulary and model name
match exactly. The lane output also records the embedding model, cache filenames,
runtime package versions, and clustering parameters.

To explain the spectral decomposition for a seiyuu:

```bash
.venv/bin/python scripts/explain_semantic_eigenvectors.py --seiyuu "Ayana Taketatsu"
```

This writes:

```text
models/semantic_character_lanes/ayana_taketatsu_spectral_explanation.json
```

For each nontrivial Laplacian eigenvector, the explanation JSON includes
positive/negative character poles and maps each pole back to descriptors by
pooling the original semantic TF-IDF descriptor weights with the eigenvector
loadings.

## Character Distance

Each character is represented as a weighted bag of semantic descriptors from the
TF-IDF matrix. For each character, the runner keeps the top weighted descriptors
and normalizes their weights.

For two characters, distance is a symmetric weighted soft nearest-neighbor
descriptor distance:

```text
D(A, B) =
  0.5 * sum_i weight_A[i] * min_j descriptor_distance(A_i, B_j)
+ 0.5 * sum_j weight_B[j] * min_i descriptor_distance(B_j, A_i)
```

This is cheaper than optimal transport, but captures the same basic idea: two
characters can be close if their descriptors have near semantic matches, even
when no descriptor string is identical.

## Lane Discovery

For one seiyuu:

1. Compute the character x character distance matrix.
2. Convert distance to affinity with:

```text
affinity_ij = exp(-distance_ij / median_nonzero_distance)
```

3. Diagonalize the normalized graph Laplacian.
4. Cluster the spectral coordinates with deterministic Ward clustering.
5. For each cluster, pool the original descriptor weights from member characters
   and rank descriptors by pooled weight.

The lane descriptors are therefore explanations of the clustered characters,
not labels invented by the embedding model.

## Global Descriptor Basis Experiment

For the LLM descriptor pipeline, each descriptor is first expanded into four
short Qwen/Ollama glosses and embedded with BGE-small. The descriptor vector is
the normalized mean of those four gloss embeddings:

```text
E_i = normalize(mean(gloss_embedding_i1 ... gloss_embedding_i4))
G = E @ E.T
```

One reproducible basis-selection experiment uses uncentered `G` and ranks
candidate basis functions by raw off-diagonal row mass:

```text
support_i = sum_j G_ij - G_ii
```

This favors descriptors that are semantically connected to many other
descriptors, rather than descriptors that are merely far from the global center
of mass. Pivoted Cholesky can then use that support score in three modes:

```bash
.venv/bin/python scripts/pivot_global_descriptor_basis.py --basis-centering none --pivot-priority row_sum
.venv/bin/python scripts/pivot_global_descriptor_basis.py --basis-centering none --pivot-priority row_sum_first
.venv/bin/python scripts/pivot_global_descriptor_basis.py --basis-centering none --pivot-priority row_sum_residual
```

The current most interpretable global ordering is `row_sum`: it walks descriptors
in descending uncentered semantic support, while skipping descriptors whose
residual has already been explained by previous pivots. This starts with broad
anchors such as `pushover`, `strict personality`, `cold demeanor`, `shy`, and
`generous`, instead of beginning from rare centered outliers.

## Ayana Taketatsu Sanity Check

With the current prototype, Nino Nakano and Kirino Kousaka land in the same
semantic lane:

```text
Ako Suminoe
Kaede Kayano
Kirino Kousaka
Kotori Itsuka
Mio Isurugi
Nino Nakano
Yuzu Aihara
```

Top pooled descriptors for that lane include:

```text
sister
tsundere
sadistic
inferiority complex
younger sister
scheming
```

Nino and Kirino are not direct nearest neighbors; their pairwise distance is
around the median of Ayana's character pairs. The cluster should be interpreted
as a broader semantic lane, not a claim that the two characters are equivalent.

## Caveats

The semantic descriptor matrix is still experimental. WordNet-backed extraction
improves alignment for phrases like `second sister` and `younger sister`, but it
also admits generic descriptors such as `girl`, `student`, and `school`.
Ranking, pruning, and UI presentation should treat these as explainable evidence
rather than final polished labels.
