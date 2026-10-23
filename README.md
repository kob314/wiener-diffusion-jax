# Wiener first-passage computations in JAX

A compact scientific Python code sample computing decision-time and reaction-time distributions for a drift-diffusion model with two absorbing boundaries.

The implementation uses Navarro–Fuss small-time and large-time density series, selecting between them using truncation-error bounds. A cumulative first-passage calculation scales the approximate decision-time bin masses. Discrete convolution adds lognormally distributed nondecision time, and a uniform component represents lapses.

## What the code demonstrates

- Array broadcasting and numerical probability calculations in JAX.
- Comparing truncation-error bounds in log space.
- Combining independent time distributions through discrete convolution.
- Visualizing response-specific distributions, choice proportions, and optional mean reaction times.

## Files

- `wiener.py`: the four numerical functions, preserved from the notebook.
- `example.py`: the plotting example, with its return value corrected to `p_after`.
- `jax-wiener-funcs.ipynb`: the original notebook content and mathematical explanations, with the same return-value fix and saved outputs cleared.
- `requirements.txt`: runtime and notebook dependencies.

## Run the example

Use Python 3.10 or newer in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python example.py
```

On Windows, activate with `.venv\Scripts\activate` instead.

The example plots A (lower-boundary) and B (upper-boundary) reaction-time bin masses, together with a stacked bar of choice percentages. Dotted lines show decision time alone; solid lines include nondecision time and lapses.

To change the optional plotting features:

```python
from example import example_wfpt

grid_t, p = example_wfpt(show_mean_RT=True, show_Wiener=True)
```

- `show_Wiener=True` overlays decision-time distributions (default: enabled).
- `show_mean_RT=True` adds full-model mean-time lines and annotations (default: disabled).

Change the model parameters inside `example_wfpt` to explore different distributions.

For the notebook:

```sh
jupyter lab jax-wiener-funcs.ipynb
```

## Parameters and outputs

| Parameter | Meaning |
| --- | --- |
| `v` | Drift rate; positive values favor the upper boundary |
| `w` | Relative starting position, between 0 and 1 |
| `a` | Positive boundary separation, with diffusion scale fixed to one |
| `t0` | Median nondecision time |
| `t0_lsig` | Standard deviation of log nondecision time |
| `l` | Lapse probability, between 0 and 1 |
| `grid_t` | Positive, uniformly spaced time column, shape `(n_times, 1)` |
| `k` | Odd number of retained series terms, for example 21 |

`jax_wfpt` and `jax_wfpt_noisy` return arrays of shape `(n_times, 2)`: column 0 is A, column 1 is B. Their entries represent approximate probability masses per time bin, although the original plotting example labels its vertical axis “probability density.”

The retained masses can sum to less than one because the time window is finite. The displayed choice percentages and means are conditional on that retained window. Shared nondecision time preserves choice proportions over an unlimited time range; truncation can change them, and uniform lapses move proportions toward equal choices.

## Scope

This repository preserves the notebook's numerical implementation rather than introducing numerical changes. It is an illustrative code sample, without automated numerical validation or parameter fitting. In particular, the zero-drift expression inside `jnp.where` can have an invalid inactive branch, and the raw series evaluations are not clipped before taking logarithms. Extreme parameters and automatic differentiation require separate validation. The functions use JAX but do not demonstrate NumPyro inference or explicit JIT compilation.

## References

- Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. *Journal of Mathematical Psychology*, 53(4), 222–230.
- Blurton, S. P., Kesselmeier, M., & Gondan, M. (2012). Fast and accurate calculations for cumulative first-passage time distributions in Wiener diffusion models. *Journal of Mathematical Psychology*, 56(6), 470–475.
