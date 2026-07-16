"""Fig 2 - Phase boundary."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

from model import (
    DEFAULT_PARAMS,
    ModelParams,
    apply_publication_style,
    save_figure,
    style_axis_text,
    style_legend,
)


DELTA_RANGE_CM = (-25.0, 25.0)
OMEGA_RANGE_CM = (0.0, 50.0)
PHASE_NUM_POINTS = 601

T_FIXED_A = [20.0, 35.0, 50.0]
COLORS_A = ["#1f77b4", "#ff7f0e", "#17becf"]
LABELS_A = [rf"$T = {t:.0f}\,^\circ$C" for t in T_FIXED_A]

def _primary_instability_omega_curve(
    delta: np.ndarray,
    temperature_c: float,
    params: ModelParams,
) -> np.ndarray:
    driving = params.alpha_T * (temperature_c - params.T0) + params.alpha_delta * delta**2
    return np.sqrt(np.maximum(driving / params.alpha_omega, 0.0))


def _structural_spinodal_omega_curve(
    delta: np.ndarray,
    temperature_c: float,
    params: ModelParams,
) -> np.ndarray:
    """Structural spinodal: solve a_m^eff = a_m - g P^2 = 0, equivalently P^2 = a_m/g.
    """
    driving = params.alpha_T * (temperature_c - params.T0) + params.alpha_delta * delta**2
    boundary_driving = driving + params.bP * params.a_m / params.g
    return np.sqrt(np.maximum(boundary_driving / params.alpha_omega, 0.0))


def set_helvetica() -> str:
    candidates = []
    direct_path = r"C:\Windows\Fonts\Helvetica.ttf"
    if os.path.exists(direct_path):
        candidates.append(direct_path)
    for font_entry in fm.fontManager.ttflist:
        if font_entry.name.lower().startswith("helvetica"):
            candidates.append(font_entry.fname)
    for path in fm.findSystemFonts():
        if "helvetica" in os.path.basename(path).lower():
            candidates.append(path)
    if not candidates:
        plt.rcParams["font.family"] = "DejaVu Sans"
        return "DejaVu Sans"

    def font_score(path):
        name = os.path.basename(path).lower()
        score = 0
        if "regular" in name or "roman" in name:
            score -= 2
        if "bold" in name or "black" in name:
            score += 2
        if "italic" in name or "oblique" in name:
            score += 1
        return (score, len(name))

    best_path = sorted(candidates, key=font_score)[0]
    fm.fontManager.addfont(best_path)
    font_name = fm.FontProperties(fname=best_path).get_name()
    plt.rcParams["font.family"] = font_name
    return font_name


def _panel_phase_boundary(ax: plt.Axes, params: ModelParams) -> None:
    delta = np.linspace(*DELTA_RANGE_CM, PHASE_NUM_POINTS)
    phase_curves: dict[float, np.ndarray] = {}
    observable_curves: dict[float, np.ndarray] = {}

    for t_fixed in T_FIXED_A:
        phase_curves[t_fixed] = _primary_instability_omega_curve(delta, t_fixed, params)
        observable_curves[t_fixed] = _structural_spinodal_omega_curve(
            delta,
            t_fixed,
            params,
        )

    reference_t = T_FIXED_A[0]
    gl_threshold = phase_curves[reference_t]
    observable_threshold = observable_curves[reference_t]
    incipient_mask = (
        (gl_threshold < OMEGA_RANGE_CM[1])
        & (observable_threshold > gl_threshold)
    )
    ax.fill_between(
        delta,
        gl_threshold,
        np.minimum(observable_threshold, OMEGA_RANGE_CM[1]),
        where=incipient_mask,
        color="#8ecae6",
        alpha=0.55,
        zorder=-2,
        label="_nolegend_",
    )
    clustered_mask = observable_threshold < OMEGA_RANGE_CM[1]
    ax.fill_between(
        delta,
        observable_threshold,
        OMEGA_RANGE_CM[1],
        where=clustered_mask,
        color="#1f77b4",
        #alpha=0.85,
        zorder=-3,
        label="_nolegend_",
    )

    for t_fixed, color, label in zip(T_FIXED_A, COLORS_A, LABELS_A):
        ax.plot(
            delta,
            phase_curves[t_fixed],
            color=color,
            linestyle="-",
            linewidth=2.1,
            label=label,
            zorder=3,
        )
        ax.plot(
            delta,
            observable_curves[t_fixed],
            color=color,
            linestyle="--",
            linewidth=2.4,
            label="_nolegend_",
            zorder=4,
        )

    ax.axvline(0.0, color="0.6", linestyle=":", linewidth=1.0)
    ax.text(
        0.50,
        0.97,
        "Structurally Ordered",
        ha="center",
        va="top",
        fontsize=16,
        color="white",
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.61,
        "Vibrationally Polarized",
        ha="center",
        va="center",
        fontsize=16,
        color="black",
        transform=ax.transAxes,
    )
    ax.text(
        0.75,
        0.12,
        "Normal Liquid",
        ha="center",
        va="center",
        fontsize=18,
        color="black",
        transform=ax.transAxes,
    )

    ax.set_xlim(*DELTA_RANGE_CM)
    ax.set_ylim(0.0, OMEGA_RANGE_CM[1])
    ax.set_xlabel(r"Detuning $\Delta$ (cm$^{-1}$)")
    ax.set_ylabel(r"Rabi splitting $\Omega$ (cm$^{-1}$)", labelpad = 10)
    legend = ax.legend(frameon=False, loc="lower left", fontsize=12)
    style_legend(legend)
    style_axis_text(ax)
    ax.tick_params(labelsize=14)
    ax.tick_params(axis="x")
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)

def main() -> None:
    apply_publication_style()

    fig, ax_phase = plt.subplots(figsize=(5.0, 4.5))

    _panel_phase_boundary(ax_phase, DEFAULT_PARAMS)

    path = save_figure(fig, "Fig2.pdf")
    plt.close(fig)
    print(f"Saved: {path.name}")


if __name__ == "__main__":
    main()
