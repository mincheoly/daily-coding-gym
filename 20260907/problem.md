## 🏋️ Technical Gym — Day 4

**Target:** ~20 minutes  
**Language:** Python + NumPy  
**Theme:** mathematical translation + numerical stability + vectorization

### Problem: Stable binary logistic loss and gradient

You are given:

```python
X.shape == (N, D)
y.shape == (N,)
w.shape == (D,)
```

where:

- `X` is a design matrix,
- `y` contains only `0.0` or `1.0`,
- `w` is a parameter vector.

Define the logits

\[
z_i = X_i \cdot w
\]

and probabilities

\[
p_i = \frac{1}{1 + e^{-z_i}}.
\]

The mean binary cross-entropy loss is

\[
L =
-\frac{1}{N}
\sum_i
\left[
y_i \log(p_i)
+
(1-y_i)\log(1-p_i)
\right].
\]

Its gradient with respect to \(w\) is

\[
\nabla_w L
=
\frac{1}{N}X^T(p-y).
\]

Implement:

```python
def logistic_loss_and_grad(X, y, w):
    ...
```

Return:

```python
loss, grad
```

where `loss` is a scalar and `grad.shape == (D,)`.

### Constraints

- No Python loops over samples or features.
- Do not use SciPy, PyTorch, sklearn, or an existing logistic-loss function.
- Your implementation must remain finite for logits with very large magnitude, including cases equivalent to:

```python
z = np.array([1000.0, -1000.0, 500.0, -500.0])
```

- Avoid expressions that produce `inf`, `nan`, or warnings internally even if the final mathematical answer would be finite.
- Assume `X`, `y`, and `w` contain finite floating-point values.
- You may use standard NumPy operations.
- Do not regularize the loss.
- Aim for **O(ND)** time and no large intermediate arrays beyond what is naturally required for the matrix-vector operations.

### Useful test case

```python
X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

y = np.array([1.0, 0.0, 1.0])
w = np.array([0.5, -0.25])
```

Also test your function with a `w` that makes some logits extremely positive or negative.

Submit whatever implementation you reach in about 20 minutes, even if you suspect part of it is numerically unstable. I’ll review **correctness, stability, vectorization, memory behavior, and clarity**.