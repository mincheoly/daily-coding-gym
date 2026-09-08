Exactly.

The target isn't **"make `exp()` never underflow."** It's:

> **Rearrange the entire computation so floating-point limitations don't contaminate the final answer.**

For example, with \(z=-1000\):

Naive:

\[
\log(1+e^{1000}) \rightarrow \log(\infty)\rightarrow\infty
\]

Stable rewrite:

\[
1000+\log(1+e^{-1000})
\]

and computationally:

\[
e^{-1000}\rightarrow0
\]

so

\[
1000+\log(1+0)=1000.
\]

We've allowed an intermediate to underflow, but it was a term so negligible that dropping it has essentially no effect on the representable answer.

This distinction—**underflow isn't inherently a bug; ask whether it damages the final quantity**—is a very useful numerical-computing instinct.

And I think you caught something important here rather than just accepting the "stable" implementation because I labeled it stable. That's exactly the kind of scrutiny we're trying to preserve with these exercises.

Day 4 is complete, so here’s **Day 5**.

Implement:

```python
def streaming_mean_var(chunks):
    ...
```

Each element of `chunks` is a 2D NumPy array of shape `(m_i, D)`, representing rows of observations with `D` features. The number of rows may differ across chunks.

Return two 1D arrays of shape `(D,)`:

- `mean`: the mean of each feature across **all rows in all chunks**
- `var`: the **sample variance** of each feature across all rows, using `ddof=1`

Your function should produce the same result, up to floating-point tolerance, as:

```python
X = np.concatenate(chunks, axis=0)
mean = X.mean(axis=0)
var = X.var(axis=0, ddof=1)
```

Constraints:

- Process the data in **one pass**
- Do **not** concatenate or retain previous chunks
- Extra memory should be `O(D)`, independent of the total number of rows
- You may use vectorized NumPy operations within each chunk
- Aim for good numerical stability when values have a large offset, e.g. data around `1e12` with relatively small variation
- Assume there are at least 2 total observations across all chunks

As before, documentation/API references are fair game, but avoid looking up a solution to the streaming-variance problem itself.