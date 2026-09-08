Exactly right on the invariance. One small correction on the stability explanation:

After subtracting the maximum, **the largest value is exactly 0**, and every other value is ≤ 0. Therefore,

\[
0 < e^{x_i-\max(x)} \leq 1
\]

so exponentiation can **never overflow**. Very negative differences can underflow toward zero, but that's generally benign here—it means that probability is negligible relative to the maximum.

Your comment about the range is also useful: the most negative shifted value can indeed be `-range(x)`. So the transformation changes something like

\[
[1000,\ 1001,\ 1002]
\]

into

\[
[-2,\ -1,\ 0],
\]

which is a vastly friendlier numerical problem.

**Day 1 complete.** ✓

And I'd stop there rather than piling on another exercise. The thing I'd want you to retain from today is simply:

> **When exponentiating a set of values that will subsequently be normalized, think about shifting by the maximum first.**

If that reflex becomes automatic again, today's 20 minutes did its job. Tomorrow at noon we'll hit a different muscle.

## 🏋️ Technical Gym — Day 2

**Target:** ~20 minutes  
**Language:** Python + NumPy  
**Theme:** numerical stability + streaming/chunking

### Problem: Streaming log-sum-exp

You receive a dataset as an **iterator of 1D NumPy arrays**. The full dataset may be far too large to fit in memory.

Implement:

```python
def streaming_logsumexp(chunks):
    ...
```

It should return the scalar

\[
\log\left(\sum_i e^{x_i}\right)
\]

over **all values in all chunks**.

For example:

```python
chunks = iter([
    np.array([1.0, 2.0]),
    np.array([3.0]),
    np.array([4.0, 5.0]),
])
```

Your function should produce the same result, up to floating-point precision, as:

```python
np.log(np.exp(np.array([1., 2., 3., 4., 5.])).sum())
```

### Constraints

- `chunks` may be a one-pass iterator; assume you **cannot rewind it**.
- You may not concatenate the chunks or store the whole dataset.
- Peak auxiliary memory should be **O(size of the largest chunk)**, not O(total dataset size).
- Do not use SciPy or an existing `logsumexp` implementation.
- The implementation must remain numerically well-behaved for data such as:

```python
iter([
    np.array([1000.0, 1001.0]),
    np.array([999.0, 1002.0]),
])
```

and:

```python
iter([
    np.array([-1000.0, -1001.0]),
    np.array([-999.0, -1002.0]),
])
```

- Assume every value is finite and that the dataset contains at least one value.
- Python loops **over chunks are allowed**. Loops over individual elements inside each chunk are not.
- Return a Python or NumPy scalar.

### Submission

Send me your implementation when you're done. I’ll review it for **correctness, numerical stability, streaming behavior, memory use, and clarity**.

No need to make it elegant before submitting—the 20-minute version is the useful one.