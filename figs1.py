"""Fig S1 - Detuning-temperature phase diagram at fixed Rabi splittings."""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from model import (
    DEFAULT_PARAMS,
    ModelParams,
    apply_publication_style,
    save_figure,
    style_axis_text,
    style_legend,
)


DELTA_RANGE_CM = (-35.0, 35.0)
TEMPERATURE_RANGE_C = (0.0, 100.0)
PHASE_NUM_POINTS = 801

OMEGA_FIXED_CM = [20.0, np.sqrt(1250.0)]
COLORS = ["#ff7f0e", "#1f77b4"]
LABELS = [
    rf"$\Omega = {OMEGA_FIXED_CM[0]:.1f}\,\mathrm{{cm}}^{{-1}}$",
    rf"$\Omega = {OMEGA_FIXED_CM[1]:.1f}\,\mathrm{{cm}}^{{-1}}$",
]


def _critical_temperature_curve(
    delta: np.ndarray,
    omega: float,
    params: ModelParams,
) -> np.ndarray:
    return params.T0 + (
        params.alpha_omega * omega**2 - params.alpha_delta * delta**2
    ) / params.alpha_T


def _rayleigh_temperature_curve(
    delta: np.ndarray,
    omega: float,
    params: ModelParams,
) -> np.ndarray:
    """Structural spinodal condition a_m^eff = 0, i.e. P^2 = a_m/g (the
    same locus where the sigmoid Rayleigh readout crosses S_R = 1/2)."""
    return _critical_temperature_curve(delta, omega, params) - (
        params.bP * params.a_m / params.g / params.alpha_T
    )


def _visible_mask(curve: np.ndarray) -> np.ndarray:
    return curve > TEMPERATURE_RANGE_C[0]


def _panel_delta_temperature(ax: plt.Axes, params: ModelParams) -> None:
    delta = np.linspace(*DELTA_RANGE_CM, PHASE_NUM_POINTS)
    phase_curves: dict[float, np.ndarray] = {}
    observable_curves: dict[float, np.ndarray] = {}

    for omega in OMEGA_FIXED_CM:
        phase_curves[omega] = _critical_temperature_curve(delta, omega, params)
        observable_curves[omega] = _rayleigh_temperature_curve(delta, omega, params)

    reference_omega = OMEGA_FIXED_CM[1]
    gl_threshold = phase_curves[reference_omega]
    observable_threshold = observable_curves[reference_omega]
    ax.fill_between(
        delta,
        np.maximum(observable_threshold, TEMPERATURE_RANGE_C[0]),
        np.minimum(gl_threshold, TEMPERATURE_RANGE_C[1]),
        where=(gl_threshold > TEMPERATURE_RANGE_C[0])
        & (observable_threshold < gl_threshold),
        color="#8ecae6",
        alpha=0.55,
        zorder=-2,
        label="_nolegend_",
    )
    ax.fill_between(
        delta,
        TEMPERATURE_RANGE_C[0],
        np.minimum(observable_threshold, TEMPERATURE_RANGE_C[1]),
        where=observable_threshold > TEMPERATURE_RANGE_C[0],
        color="#1f77b4",
        alpha=0.90,
        zorder=-3,
        label="_nolegend_",
    )

    for omega, color, label in zip(OMEGA_FIXED_CM, COLORS, LABELS):
        gl_curve = phase_curves[omega]
        rayleigh_curve = observable_curves[omega]
        ax.plot(
            delta,
            gl_curve,
            color=color,
            linestyle="-",
            linewidth=2.1,
            label=label,
            zorder=3,
        )
        if np.any(_visible_mask(rayleigh_curve)):
            ax.plot(
                delta,
                rayleigh_curve,
                color=color,
                linestyle="--",
                linewidth=2.4,
                label="_nolegend_",
                zorder=4,
            )

    ax.axvline(0.0, color="0.6", linestyle=":", linewidth=1.0)
    ax.text(
        0.50,
        0.08,
        "Structurally"
        "\n"
        "Ordered",
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.6,
        color="white",
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.64,
        "Vibrationally"
        "\n"
        "Polarized",
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.6,
        color="black",
        transform=ax.transAxes,
    )
    ax.text(
        0.12,
        0.88,
        "Normal"
        "\n"
        "Liquid",
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.6,
        color="black",
        transform=ax.transAxes,
    )

    ax.set_xlim(*DELTA_RANGE_CM)
    ax.set_ylim(*TEMPERATURE_RANGE_C)
    ax.set_xlabel(r"Detuning $\Delta$ (cm$^{-1}$)")
    ax.set_ylabel(r"Temperature $T$ ($^\circ$C)", labelpad=10)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=11)
    style_legend(legend)
    style_axis_text(ax)
    ax.tick_params(labelsize=14)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)


def main() -> None:
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    _panel_delta_temperature(ax, DEFAULT_PARAMS)

    path = save_figure(fig, "FigS1.pdf")
    plt.close(fig)
    sys.stdout.buffer.write(f"Saved: {path.name}\n".encode())


if __name__ == "__main__":
    main()
