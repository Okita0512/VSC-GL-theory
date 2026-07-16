"""Fig S2 - Robustness of the thermal-collapse readout to the sigmoid width sigma_a.

Repeats the Fig. 3b thermal-collapse scan (Delta = 0, Omega^2 = 1400 cm^-2) while
varying only the Rayleigh readout width sigma_a = rayleigh_width_a. All other GL
parameters (DEFAULT_PARAMS) are held fixed, so the thermodynamic trajectories
a_P(T) and a_m^eff(T), and hence the structural spinodal a_m^eff = 0, are
unchanged; only the sharpness of the optical crossover S_R(a_m^eff) varies.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

from model import (
    DEFAULT_PARAMS,
    apply_publication_style,
    rayleigh_enhancement_ratio,
    rayleigh_transition_temperature,
    save_figure,
    style_axis_text,
    style_legend,
)


DELTA_CM = 0.0
OMEGA_SQ_CM2 = 1400.0
TEMPERATURE_SCAN_C = (10.0, 85.0)
NUM_POINTS = 801

# Representative sigma_a values spanning a factor of 5 around the default.
SIGMA_A_SPECS = [
    (0.010, "0.010"),
    (DEFAULT_PARAMS.rayleigh_width_a, "0.0215"),
    (0.050, "0.050"),
]
COLORS = ["#2ca02c", "#1f77b4", "#d62728"]


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


def _normalized_rayleigh_enhancement(
    omega_r: float,
    temperatures: np.ndarray,
    sigma_a: float,
) -> np.ndarray:
    params = dataclasses.replace(DEFAULT_PARAMS, rayleigh_width_a=sigma_a)
    intensity = rayleigh_enhancement_ratio(
        delta_cm=DELTA_CM,
        omega_r_cm=omega_r,
        temperature_c=temperatures,
        params=params,
    )
    excess = np.clip(intensity - 1.0, 0.0, None)
    normalization = max(float(excess.max()), 1.0e-12)
    return excess / normalization


def main() -> None:
    apply_publication_style()

    temperatures = np.linspace(*TEMPERATURE_SCAN_C, NUM_POINTS)
    omega_r = float(np.sqrt(OMEGA_SQ_CM2))

    # a_m^eff = 0 does not depend on sigma_a (readout-only parameter).
    t_spinodal = float(
        rayleigh_transition_temperature(
            omega_r_cm=omega_r,
            delta_cm=DELTA_CM,
            params=DEFAULT_PARAMS,
        )
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for (sigma_a, label_value), color in zip(SIGMA_A_SPECS, COLORS):
        normalized = _normalized_rayleigh_enhancement(omega_r, temperatures, sigma_a)
        is_default = np.isclose(sigma_a, DEFAULT_PARAMS.rayleigh_width_a)
        label = rf"$\sigma_a = {label_value}$" + (" (default)" if is_default else "")
        ax.plot(temperatures, normalized, color=color, linewidth=2.2, label=label)

    ax.axvline(
        t_spinodal,
        color="0.25",
        linestyle="--",
        linewidth=1.5,
    )
    ax.text(
        t_spinodal + 1.8,
        0.70,
        r"$a_m^{\rm eff}=0$ (structural spinodal)",
        color="0.25",
        rotation=0,
        ha="left",
        va="center",
        fontsize=12,
        transform=ax.get_xaxis_transform(),
    )

    ax.text(
        0.97,
        0.92,
        r"$\Delta = 0$, $\Omega^2 = 1400\,\mathrm{cm}^{-2}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

    ax.set_xlabel(r"Temperature $T$ ($^\circ$C)")
    ax.set_ylabel("Normalized Rayleigh enhancement")
    ax.set_xlim(*TEMPERATURE_SCAN_C)
    ax.set_ylim(-0.05, 1.19)
    legend = ax.legend(frameon=False, loc="lower left", fontsize=10)
    style_legend(legend)
    style_axis_text(ax)
    ax.tick_params(labelsize=14)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)

    path = save_figure(fig, "FigS2.pdf")
    plt.close(fig)
    sys.stdout.buffer.write(
        (f"Saved: {path.name}\n"
         f"T_spinodal (a_m_eff=0, independent of sigma_a) = {t_spinodal:.4f} C\n").encode()
    )


if __name__ == "__main__":
    main()
