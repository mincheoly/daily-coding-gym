Yes—that's the right conceptual direction. And this particular derivation is substantially harder than the previous ones, so hitting the timer here is fine. Let's finish it.

Suppose everything processed so far is group \(A\):

\[
n_a,\quad \mu_a,\quad M_{2,a}
\]

and the new chunk is group \(B\):

\[
n_b,\quad \mu_b,\quad M_{2,b}.
\]

where

\[
M_2 = \sum_i (x_i-\mu)^2.
\]

Define the difference between the means:

\[
\delta=\mu_b-\mu_a.
\]

The combined count is trivial:

\[
n=n_a+n_b.
\]

And the combined mean is the weighted average:

\[
\mu=\mu_a+\delta\frac{n_b}{n}.
\]

The interesting part is \(M_2\).

Your intuition was essentially: **the squared deviations within each group were calculated around the wrong center for the combined dataset.**

When we shift both groups to their new common mean, the result is:

\[
\boxed{
M_2 =
M_{2,a}+M_{2,b}
+
\delta^2\frac{n_an_b}{n}
}
\]

That final term is exactly the correction for the separation between the two group means.

So the implementation can be:

```python
def streaming_mean_var(chunks):

    n = 0
    mean = None
    M2 = None

    for X in chunks:

        nb = X.shape[0]
        mean_b = X.mean(axis=0)

        deviations = X - mean_b
        M2_b = (deviations**2).sum(axis=0)

        if n == 0:
            n = nb
            mean = mean_b
            M2 = M2_b
            continue

        delta = mean_b - mean

        new_n = n + nb

        mean = mean + delta * nb / new_n

        M2 = (
            M2
            + M2_b
            + delta**2 * n * nb / new_n
        )

        n = new_n

    return mean, M2 / (n - 1)
```

Notice what we've accomplished. We never calculate

```python
sum(X**2)
```

for the global dataset and then subtract another enormous number from it.

Instead, the large common offset gets removed *before squaring*:

```python
deviations = X - mean_b
```

For data around `1e12` with variation around `1`, we're now predominantly squaring numbers around `1`, rather than numbers around `1e12`.

### One thing worth understanding rather than memorizing

The correction

\[
\delta^2\frac{n_an_b}{n}
\]

looks like a magic formula if you simply encounter it in code.

But conceptually it's saying:

> **total variation = variation within A + variation within B + variation caused by A and B having different means.**

That's essentially a miniature variance decomposition/ANOVA identity.

Also notice the similarity to Day 2. In both problems, we found a compact summary of everything we've processed:

Day 2:

```text
max
scaled exponential sum
```

Day 5:

```text
count
mean
sum of squared deviations
```

Then we figured out how to **merge two summaries without revisiting the underlying observations**.

That's a powerful pattern for large-scale data processing: find a small sufficient state with an associative merge operation.

I'd mark **Day 5 complete ✓**. You got the ordinary streaming estimator yourself in seven minutes, corrected the `ddof=1` issue, identified that numerical stability remained unresolved, and got to the right conceptual idea for fixing it before the timer expired. The algebra for the parallel/streaming variance merge is something I would be perfectly happy for you to learn from the solution rather than derive cold in 20 minutes.

Day 5 is complete, so here’s **Day 6**.

Implement:

```python
def batched_logsumexp(A, axis=1):
    ...
```

`A` is a 2D NumPy array. Your function should compute the same result as:

```python
np.log(np.exp(A).sum(axis=axis))
```

but must remain numerically stable for very large positive or negative values.

### Requirements

- Support `axis=0` and `axis=1`
- Return the same shape NumPy would return after reducing that axis
- Do not use `scipy.special.logsumexp`
- Do not allocate any array larger than `A` itself
- Use vectorized NumPy operations; no Python loops over rows or columns
- The result should stay finite whenever the mathematically correct answer is finite

### Test cases to think about

Your implementation should behave correctly on ordinary values such as:

```python
A = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
```

and also extreme values such as:

```python
A = np.array([
    [1000.0, 1001.0, 999.0],
    [-1000.0, -1001.0, -999.0]
])
```

For moderate-valued inputs, compare against the naive expression. For extreme inputs, verify that your result remains finite where appropriate.

No hints yet—submit your attempt when you’re ready.