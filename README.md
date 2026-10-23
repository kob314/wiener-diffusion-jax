# Wiener first-passage computations in JAX

Computing decision-time and reaction-time distributions for a Wiener drift-diffusion model with two absorbing boundaries.

The implementation uses Navarro–Fuss small-time and large-time density series, selecting between them using truncation-error bounds. A cumulative first-passage calculation scales the approximate decision-time bin masses. Discrete convolution adds lognormally distributed nondecision time, and a uniform component represents lapses.

## Interactive version

Run the code within Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xn1Y7ihZyPaxQRz19Ej4XFioDPMOFnSP?usp=sharing)

## Files

- `wiener.py`: four numerical functions
- `example.py`: the plotting example
- `jax-wiener-funcs.ipynb`: the functions above organized into a notebook 
- `requirements.txt`: runtime and notebook dependencies.


## Figure
The example plots A (lower-boundary) and B (upper-boundary) reaction-time bin masses, together with a stacked bar of choice percentages. Dotted lines show decision time alone; solid lines include nondecision time and lapses.

To change the optional plotting features:

```python
from example import example_wfpt

grid_t, p = example_wfpt(show_mean_RT=True, show_Wiener=True)
```

- `show_Wiener=True` overlays decision-time distributions (default: enabled).
- `show_mean_RT=True` adds full-model mean-time lines and annotations (default: disabled).

Change the model parameters inside `example_wfpt` to explore different distributions.


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


## References

- Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. *Journal of Mathematical Psychology*, 53(4), 222–230.
- Blurton, S. P., Kesselmeier, M., & Gondan, M. (2012). Fast and accurate calculations for cumulative first-passage time distributions in Wiener diffusion models. *Journal of Mathematical Psychology*, 56(6), 470–475.
