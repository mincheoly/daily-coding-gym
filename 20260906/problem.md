## 🏋️ Technical Gym — Day 3

**Target:** ~20 minutes  
**Language:** Python + NumPy  
**Theme:** mathematical translation + vectorization + memory awareness

### Problem: Pairwise squared Euclidean distances

You are given two NumPy arrays:

```python
X.shape == (N, D)
Y.shape == (M, D)
```

Implement:

```python
def pairwise_sq_distances(X, Y):
    ...
```

Return an array `D2` of shape `(N, M)` such that

\[
D2[i,j] = \sum_k (X[i,k] - Y[j,k])^2
\]

For example:

```python
X = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
])

Y = np.array([
    [0.0, 1.0],
    [2.0, 0.0],
])
```

should produce approximately:

```python
array([
    [1.0, 4.0],
    [2.0, 1.0],
])
```

### Constraints

- Do **not** use Python loops over rows or coordinates.
- Do **not** construct an intermediate array of shape `(N, M, D)`.
- The final `(N, M)` result is unavoidable, but auxiliary memory should scale roughly as **O(N + M + NM)** rather than **O(NMD)**.
- Do not use SciPy or an existing pairwise-distance function.
- Assume `X` and `Y` are nonempty 2D floating-point arrays with the same number of columns.
- Your result should not contain negative distances, including tiny negative values that might arise from floating-point arithmetic.
- Aim for behavior that remains reasonable when coordinates have substantially different magnitudes, e.g.:

```python
X = np.array([
    [10000.0, 0.001],
    [-10000.0, 0.002],
])

Y = np.array([
    [9999.0, 0.003],
    [0.0, 0.0],
])
```

### Submission

Send me the implementation you reach in about 20 minutes, even if you think it has problems.

I’ll review it for **correctness, vectorization, memory behavior, numerical issues, and clarity**.