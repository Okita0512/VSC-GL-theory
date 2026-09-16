"""TOC schematic: effective Landau free-energy softening of structural order."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# Landau parameters for a schematic effective free-energy density:
# f_eff(m) = 1/2 a_m^eff m^2 + 1/4 b_m^eff m^4, with b_m^eff > 0.
B_EFF_QUARTIC = 1.0
A_EFF_VALUES = {
    r"$a_m^{\rm eff}>0$": 1.0,
    r"$a_m^{\rm eff}=0$": 0.0,
    r"$a_m^{\rm eff}<0$": -1.0,
}

M_RANGE = (0.0, 2.2)
NUM_POINTS = 1000

FIGSIZE = (3.2, 3.0)
PNG_DPI = 600


def effective_free_energy_density(
    m: np.ndarray | float, a_eff: float, b_eff: float
) -> np.ndarray | float:
    """Return the effective quartic free-energy density for structural order."""
    return 0.5 * a_eff * m**2 + 0.25 * b_eff * m**4


def minimum_positions(a_eff: float, b_eff: float) -> list[float]:
    """Analytic minimum positions of the schematic Landau landscape."""
    if a_eff < 0.0:
        m0 = float(np.sqrt(-a_eff / b_eff))
        return [m0]
    return [0.0]


def apply_manuscript_style() -> None:
    """Set a compact, clean style suitable for a manuscript panel."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def main() -> None:
    apply_manuscript_style()

    m = np.linspace(*M_RANGE, NUM_POINTS)

    curve_styles = {
        1.0: {"color": "#557a95", "lw": 2.7, "ls": "-"},
        0.0: {"color": "#6f6f6f", "lw": 2.5, "ls": "--"},
        -1.0: {"color": "#b85c5c", "lw": 2.7, "ls": "-"},
    }

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, a_eff in A_EFF_VALUES.items():
        # No vertical shift is applied here: this is the standard Landau
        # schematic with f_eff(0) = 0 for every value of a_m^eff.
        f_landau = effective_free_energy_density(m, a_eff, B_EFF_QUARTIC)
        ax.plot(m, f_landau, label=label, **curve_styles[a_eff])

        for m_min in minimum_positions(a_eff, B_EFF_QUARTIC):
            f_min = effective_free_energy_density(m_min, a_eff, B_EFF_QUARTIC)
            ax.plot(
                m_min,
                f_min,
                "o",
                ms=4.0,
                color=curve_styles[a_eff]["color"],
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=5,
            )

    ax.text(
        0.35,
        0.45,
        r"$a_m^{\rm eff}>0$",
        color="#425f76",
        ha="left",
        va="center",
        fontsize=10,
    )
    ax.text(
        1.0,
        0.12,
        r"$a_m^{\rm eff}=0$",
        color="#4f4f4f",
        ha="left",
        va="center",
        fontsize=10,
    )
    ax.text(
        1.42,
        -0.16,
        r"$a_m^{\rm eff}<0$",
        color="#8f4141",
        ha="left",
        va="center",
        fontsize=10,
    )

    ax.text(
        0.64,
        -0.45,
        r"$f_{\rm eff}(m)=\frac{a_m^{\rm eff}}{2} m^2 + \frac{b_m^{\rm eff}}{4} m^4$",
        color="#333333",
        ha="left",
        va="center",
        fontsize=10,
    )

    # ax.annotate(
    #     r"increasing $\Omega$",
    #     xy=(2.02, 0.28),
    #     xytext=(0.82, 1.24),
    #     xycoords="data",
    #     textcoords="data",
    #     arrowprops={
    #         "arrowstyle": "->",
    #         "lw": 0.8,
    #         "color": "#333333",
    #         "shrinkA": 2,
    #         "shrinkB": 1,
    #     },
    #     color="#333333",
    #     ha="center",
    #     va="center",
    #     fontsize=7.1,
    # )

    ax.set_xlabel("")
    # ax.set_ylabel(r"$f_{\rm eff}(m)$", fontsize=16)
    ax.set_xlim(0.0, 2.25)
    ax.set_ylim(-0.34, 1.6)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.grid(False)

    axis_arrow = {
        "arrowstyle": "-|>",
        "lw": 0.9,
        "color": "black",
        "mutation_scale": 8.5,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate(
        "",
        xy=(2.25, 0.0),
        xytext=(0.0, 0.0),
        arrowprops=axis_arrow,
        clip_on=False,
        zorder=1,
    )
    ax.text(
        2.10,
        -0.07,
        r"$m$",
        color="black",
        ha="center",
        va="top",
        fontsize=16,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.6),
        xytext=(0.0, -0.34),
        arrowprops=axis_arrow,
        clip_on=False,
        zorder=1,
    )

    fig.tight_layout(pad=0.25)
    fig.savefig("GL_free_energy.pdf")
    # fig.savefig("fig1b_GL_free_energy.png", dpi=PNG_DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
