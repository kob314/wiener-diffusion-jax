import jax.numpy as jnp
from wiener import jax_wfpt, jax_wfpt_noisy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def example_wfpt(show_mean_RT=False, show_Wiener=True):
    """Plot reaction-time distributions and the corresponding choice shares."""

    grid_t = jnp.linspace(0.01, 3.0, 300).reshape(-1, 1)



    grid_t = jnp.linspace(0.01, 3.0, 300).reshape(-1, 1)
    v, w, a = 1.0, 0.4, 1.5
    t0, t0_lsig, lapse = 0.3, 0.2, 0.02
    k = 21

    p_before = np.asarray(jax_wfpt(v, w, a, grid_t, k))
    p_after = np.asarray(
        jax_wfpt_noisy(v, w, a, t0, t0_lsig, lapse, grid_t, k)
    )

    # Choice percentages conditional on the retained reaction-time window.
    choice_mass = p_after.sum(axis=0)
    percentages = 100.0 * choice_mass / choice_mass.sum()
    pct_a, pct_b = map(float, percentages)

    colors = ("tab:blue", "tab:orange")
    fig, (ax_rt, ax_choice) = plt.subplots(1, 2, figsize=(8, 4),
        gridspec_kw={"width_ratios": [5, 1]},
        layout="constrained",
    )

    if show_Wiener:
        ax_rt.plot(grid_t[:, 0], p_before[:, 0], color=tuple(c + 0.5 * (1 - c) for c in to_rgb(colors[0])), label="resp. A (Wiener only)", linestyle=':')
        ax_rt.plot(grid_t[:, 0], p_before[:, 1], color=tuple(c + 0.5 * (1 - c) for c in to_rgb(colors[1])), label="resp. B (Wiener only)", linestyle=':')
    ax_rt.plot(grid_t[:, 0], p_after[:, 0], color=colors[0], label="resp. A (full model)")
    ax_rt.plot(grid_t[:, 0], p_after[:, 1], color=colors[1], label="resp. B (full model)")
    ax_rt.set_xlabel("reaction time (s)")
    ax_rt.set_ylabel("probability density")
    ax_rt.legend()

    # Mean RT for each response, conditional on the retained time window.
    if show_mean_RT:
        mean_rt = (np.asarray(grid_t) * p_after).sum(axis=0) / choice_mass
    
        for i, mean in enumerate(mean_rt):
            ax_rt.axvline(mean, color=colors[i], linewidth=1, alpha=0.7)
            ax_rt.annotate(
                f"mean RT = {mean:.2f} s",
                xy=(mean, 0.55 - 0.15 * i),
                xycoords=ax_rt.get_xaxis_transform(),
                xytext=(5, 0),
                textcoords="offset points",
                color=colors[i],
                rotation=90,
                va="center",
            )


    # A single stacked bar summarizes the two choice percentages.
    ax_choice.bar(0, pct_a, width=0.6, color=colors[0])
    ax_choice.bar(0, pct_b, bottom=pct_a, width=0.6, color=colors[1])

    ax_choice.text(
        0, pct_a / 2, f"A\n{pct_a:.1f}%",
        ha="center", va="center", color="white",
    )
    ax_choice.text(
        0, pct_a + pct_b / 2, f"B\n{pct_b:.1f}%",
        ha="center", va="center", color="white",
    )
    ax_choice.set_ylim(0, 100)
    ax_choice.set_xlim(-0.6, 0.6)
    ax_choice.set_axis_off()
    ax_choice.set_title("proportion\nof choices", loc="center")


    plt.show()
    return grid_t, p_after


if __name__ == "__main__":
    grid_t, p = example_wfpt()
