Day 6 is complete, so here’s **Day 7**.

Implement:

```python
def nearest_neighbor_distances(X, Y, chunk_size=1024):
    ...
```

`X` has shape `(N, D)` and `Y` has shape `(M, D)`. For each row `x_i` in `X`, return the Euclidean distance to its nearest row in `Y`.

The output should be a 1D NumPy array of shape `(N,)`, equivalent to:

```python
distances = np.sqrt(((X[:, None, :] - Y[None, :, :])**2).sum(axis=2))
result = distances.min(axis=1)
```

but without constructing the full `(N, M, D)` or `(N, M)` arrays.

Requirements:

- Use NumPy.
- Process `X` in chunks of at most `chunk_size` rows.
- Do not use Python loops over individual rows of `X` or `Y`.
- A Python loop over chunks is allowed.
- Extra memory should be `O(chunk_size * M)` or better, not `O(N * M)`.
- Return results in the original order of `X`.
- Handle `N`, `M`, and `D` being arbitrary positive integers.
- Your result should match the naive implementation to floating-point tolerance on moderate-sized test cases.
- Avoid unnecessary temporary arrays where practical.

Assume ordinary finite `float64` inputs; this one is primarily about **vectorization + memory-aware chunking**, not numerical-stability edge cases.

No hints or solution unless you submit an attempt or ask for one.

Your task did not run because you have reached a usage limit.