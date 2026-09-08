Absolutely. 😄 Day 1 should give me a baseline, so I won't make it artificially difficult.

## 🏋️ Technical Gym — Day 1

**Target:** ~20 minutes  
**Language:** Python + NumPy  
**AI/code completion:** ideally off until you submit  
**Theme:** mathematical implementation → numerical stability

### Problem: Stable row-wise softmax

You are given a 2D NumPy array `X` of shape `(N, D)` containing floating-point values.

Implement:

```python
def softmax_rows(X):
    ...
```

For every row \(x\), return

\[
p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
\]

Your output must:

1. Have the same shape as `X`.
2. Compute softmax independently for each row.
3. **Not use** `scipy`, `torch`, or an existing softmax/logsumexp function.
4. Be vectorized: **no Python loops over rows or columns.**
5. Work correctly for ordinary inputs such as:

```python
X = np.array([
    [1.0, 2.0, 3.0],
    [0.0, 0.0, 0.0],
])
```

6. Remain numerically well-behaved for inputs such as:

```python
X = np.array([
    [1000.0, 1001.0, 1002.0],
    [-1000.0, -1001.0, -1002.0],
])
```

Assume `X` is nonempty and contains finite floating-point values.

### Your submission

Send me **only your implementation at first**—whatever you manage within ~20 minutes. Don't worry about making it beautiful.

I'll review it on four dimensions: **correctness, numerical stability, vectorization/efficiency, and code quality.**

Then I'll give you feedback and, if appropriate, a follow-up constraint that makes the same problem slightly more interesting.

**Timer starts when you start coding, not when you read this message.**