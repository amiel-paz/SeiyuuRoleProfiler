# Antipodal Descriptor Basis Experiment

Once safe enrichment and LLM tagging finish, try a descriptor basis built from
real descriptor embeddings rather than opaque latent vectors.

## Input

- `V`: normalized descriptor embeddings, shape `384 x N`.
- `G = V.T @ V`: descriptor cosine-overlap matrix, computed fully or in blocks.
- Candidate partner count: `k = 16, 32, or 64`.

## Candidate Pairs

For each descriptor vector `v_i`, keep its `k` most negative-overlap partners:

```text
j in argsort(G_ij)[:k]
```

Deduplicate unordered pairs `(i, j)`. For `N = 10000`, this gives roughly:

```text
k = 32 -> 160k candidate pairs
k = 64 -> 320k candidate pairs
```

For each pair:

```text
delta_p = ||v_i + v_j||^2 = 2(1 + G_ij)
d_p = normalize(v_i - v_j)
```

`delta_p` measures antipodal defect. Good pairs have small `delta_p`, or
equivalently large `-G_ij`.

## Greedy Selection

Select disjoint pairs by max-det gain over the pair axes `d_p`, penalizing poor
antipodal quality:

```text
score_p = logdet_gain(d_p | selected_axes) - lambda * delta_p
```

Equivalently, track the residual after projecting out the selected axes:

```text
r_p = ||(I - P_selected) d_p||^2
```

Then `r_p` is the geometric "new direction" score. A pair is useful only if it
both adds a new direction and has real opposite poles.

## Stopping Rules

Do not force 384 axes if the descriptor pool does not support them. Stop when
any of the following fail:

```text
max_p r_p >= tau_axis
max_p (-G_ij) >= tau_anti
max_p r_p * (-G_ij) >= tau_joint
```

Initial values to try:

```text
tau_axis = 0.9
tau_anti = 0.25
tau_joint = 0.2
```

These thresholds are diagnostics, not doctrine. If BGE descriptor space is
anisotropic and true antipodes are rare, the result should be fewer axes rather
than fake coverage.

## Diagnostics

Report:

- selected axis count `A_eff`
- selected pair labels and cosine overlaps
- rank of selected axes
- condition number of `D_selected @ D_selected.T`
- positive-spanning feasibility for selected original poles:

```text
S = [v_i1, v_j1, ..., v_iA, v_jA]
find w > eps:
  S w = 0
  sum(w) = 1
```

- descriptor-pool coverage quantiles:

```text
max_a |d_a.T @ v|
```

This gives an interpretable, data-derived semantic basis anchored by actual
descriptor phrases.
