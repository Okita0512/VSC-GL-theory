"""Fig. 3 uses the direct Rayleigh readout of main-text Eqs. 5-6.

All plotted theory curves are normalized excess intensities, so the absolute
Rayleigh contrast scale I0 cancels.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
DELTA_OFF_CM = 35.0
OFF_RESONANCE_TEMPERATURE_C = 0.0
TEMPERATURE_SCAN_C = (10.0, 85.0)
TEMPERATURE_VALUES_C = [0.0, 10.0, 20.0, 30.0]
OMEGA_SQ_VALUES_CM2 = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
PANEL_B_OMEGA_SQ_CM2 = 1400.0
PANEL_A_EXPERIMENT_DATA_PATH = (
    "exp_data/figure5A_toluene_benzeneD6_digitized.csv"
)
PANEL_B_EXPERIMENT_DATA_PATH = "exp_data/figure5B_temperature_digitized.txt"
OMEGA_SCAN_CM = (0.0, 50.0)
NUM_POINTS = 801


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


def _rayleigh_enhancement_omega(
    omega_r_scan: np.ndarray,
    temperature: float,
    center_delta_cm: float = DELTA_CM,
) -> np.ndarray:
    """Rayleigh enhancement ratio from the direct readout expression."""
    return rayleigh_enhancement_ratio(
        delta_cm=center_delta_cm,
        omega_r_cm=omega_r_scan,
        temperature_c=temperature,
        params=DEFAULT_PARAMS,
    )


def _rayleigh_enhancement_T(
    omega_r: float,
    temperatures: np.ndarray,
    center_delta_cm: float = DELTA_CM,
) -> np.ndarray:
    """Rayleigh enhancement ratio from the direct readout expression vs T."""
    return rayleigh_enhancement_ratio(
        delta_cm=center_delta_cm,
        omega_r_cm=omega_r,
        temperature_c=temperatures,
        params=DEFAULT_PARAMS,
    )


def main() -> None:
    apply_publication_style()

    temperatures = np.linspace(*TEMPERATURE_SCAN_C, NUM_POINTS)
    omega_r_scan = np.linspace(*OMEGA_SCAN_CM, NUM_POINTS)
    omega_sq_scan = omega_r_scan**2
    omega_r_values = np.sqrt(np.asarray(OMEGA_SQ_VALUES_CM2, dtype=float))
    panel_b_omega_r = float(np.sqrt(PANEL_B_OMEGA_SQ_CM2))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#dd8452"]

    fig, (ax_abs, ax_red) = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.32},
    )

    omega_sq_values = []
    transition_values = []

    # Left panel: direct Eq. 5-6 Rayleigh readout vs Omega^2 at fixed T.
    reference_excess = np.clip(
        _rayleigh_enhancement_omega(
            omega_r_scan,
            TEMPERATURE_VALUES_C[0],
            center_delta_cm=DELTA_CM,
        ) - 1.0,
        0.0,
        None,
    )
    normalization = max(float(reference_excess.max()), 1.0e-12)

    intensity_off_resonance = _rayleigh_enhancement_omega(
        omega_r_scan,
        OFF_RESONANCE_TEMPERATURE_C,
        center_delta_cm=DELTA_OFF_CM,
    )
    normalized_off_resonance = (
        np.clip(intensity_off_resonance - 1.0, 0.0, None) / normalization
    )
    theory_handles = []
    (off_resonance_line,) = ax_abs.plot(
        omega_sq_scan,
        normalized_off_resonance,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=rf"Off-res., $\Delta={DELTA_OFF_CM:.0f}$ cm$^{{-1}}$, $T=0^\circ$C",
    )
    theory_handles.append(off_resonance_line)

    for color, temperature_c in zip(colors, TEMPERATURE_VALUES_C):
        intensity_vs_omega = _rayleigh_enhancement_omega(
            omega_r_scan,
            temperature_c,
            center_delta_cm=DELTA_CM,
        )
        excess_vs_omega = np.clip(intensity_vs_omega - 1.0, 0.0, None)
        normalized_vs_omega = excess_vs_omega / normalization

        (temperature_line,) = ax_abs.plot(
            omega_sq_scan,
            normalized_vs_omega,
            color=color,
            label=rf"$T = {temperature_c:.0f}^\circ$C",
        )
        theory_handles.append(temperature_line)

    # Map composition with only the blue experimental Omega_R^2 markers.
    # At shared compositions np.interp returns the measured blue value exactly;
    # otherwise it performs the required piecewise-linear interpolation.
    panel_a_data_path = Path(__file__).resolve().parent / PANEL_A_EXPERIMENT_DATA_PATH
    if not panel_a_data_path.exists():
        raise FileNotFoundError(
            f"Required panel-a experimental data not found: {panel_a_data_path}"
        )
    panel_a_data = np.genfromtxt(
        panel_a_data_path,
        delimiter=",",
        names=True,
        dtype=float,
    )
    rabi_mask = np.isfinite(panel_a_data["rabi_splitting_sq_cm_minus2"])
    rabi_compositions = panel_a_data["toluene_vol_percent"][rabi_mask]
    rabi_omega_sq = panel_a_data["rabi_splitting_sq_cm_minus2"][rabi_mask]
    rabi_order = np.argsort(rabi_compositions)
    rabi_compositions = rabi_compositions[rabi_order]
    rabi_omega_sq = rabi_omega_sq[rabi_order]

    on_mask = np.isfinite(panel_a_data["rayleigh_on_raw"])
    off_mask = np.isfinite(panel_a_data["rayleigh_off_raw"])
    rayleigh_compositions = panel_a_data["toluene_vol_percent"]
    mapped_compositions = np.concatenate(
        (rayleigh_compositions[on_mask], rayleigh_compositions[off_mask])
    )
    if (
        mapped_compositions.min() < rabi_compositions.min()
        or mapped_compositions.max() > rabi_compositions.max()
    ):
        raise ValueError("Rayleigh compositions fall outside the blue-data range.")

    omega_sq_on = np.interp(
        rayleigh_compositions[on_mask], rabi_compositions, rabi_omega_sq
    )
    omega_sq_off = np.interp(
        rayleigh_compositions[off_mask], rabi_compositions, rabi_omega_sq
    )
    # Match the maximum-normalized Figure 5B experimental convention: divide
    # both raw series by max(on resonance), with no background subtraction.
    experimental_normalization = float(
        np.nanmax(panel_a_data["rayleigh_on_raw"][on_mask])
    )
    normalized_on_experiment = (
        panel_a_data["rayleigh_on_raw"][on_mask] / experimental_normalization
    )
    normalized_off_experiment = (
        panel_a_data["rayleigh_off_raw"][off_mask] / experimental_normalization
    )
    on_experiment = ax_abs.scatter(
        omega_sq_on,
        normalized_on_experiment,
        marker="o",
        s=42,
        facecolors="none",
        edgecolors=colors[2],
        linewidth=1.2,
        zorder=6,
        label="Experiment, on-res.",
    )
    off_experiment = ax_abs.scatter(
        omega_sq_off,
        normalized_off_experiment,
        marker="D",
        s=34,
        facecolors="none",
        edgecolors="black",
        linewidth=1.2,
        zorder=6,
        label="Experiment, off-res.",
    )

    # Right panel: direct Eq. 5-6 Rayleigh readout vs T at fixed Omega^2.
    panel_b_intensity = _rayleigh_enhancement_T(
        panel_b_omega_r,
        temperatures,
        center_delta_cm=DELTA_CM,
    )
    panel_b_excess = np.clip(panel_b_intensity - 1.0, 0.0, None)
    panel_b_normalization = max(float(panel_b_excess.max()), 1.0e-12)
    panel_b_normalized = panel_b_excess / panel_b_normalization
    ax_red.plot(
        temperatures,
        panel_b_normalized,
        color=colors[-1],
        label=rf"$\Omega^2 = {PANEL_B_OMEGA_SQ_CM2:.0f}$ cm$^{{-2}}$",
    )
    data_path = Path(__file__).resolve().parent / PANEL_B_EXPERIMENT_DATA_PATH
    if data_path.exists():
        data = np.loadtxt(data_path)
        T_exp = data[:, 0]
        y_lower = data[:, 1]
        y_center = data[:, 2]
        y_upper = data[:, 3]
        yerr = [y_center - y_lower, y_upper - y_center]
        ax_red.errorbar(
            T_exp,
            y_center,
            yerr=yerr,
            fmt="s",
            color="black",
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            markersize=5,
            linestyle="none",
            label="Experiment",
        )
    else:
        print(f"Warning: {data_path} not found; skipping experimental overlay.")
    transition_temperature_panel_b = float(
        rayleigh_transition_temperature(
            panel_b_omega_r,
            delta_cm=DELTA_CM,
            params=DEFAULT_PARAMS,
        )
    )
    ax_red.axvline(
        transition_temperature_panel_b,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    # Inset: structural-spinodal transition temperature a_m^eff(T_R)=0.
    for omega_r, omega_sq in zip(omega_r_values, OMEGA_SQ_VALUES_CM2):
        omega_sq_values.append(omega_sq)
        transition_temperature = rayleigh_transition_temperature(
            omega_r,
            delta_cm=DELTA_CM,
            params=DEFAULT_PARAMS,
        )
        transition_values.append(transition_temperature)

    ax_abs.set_xlabel(r"Rabi splitting squared $\Omega^2$ (cm$^{-2}$)")
    ax_abs.set_ylabel("Normalized Rayleigh enhancement")
    ax_abs.set_xlim(700.0, 1850.0)
    ax_abs.set_ylim(-0.05, 1.19)
    legend = ax_abs.legend(
        handles=theory_handles,
        frameon=False,
        loc="upper left",
        fontsize=7.5,
        handlelength=2.3,
    )
    ax_abs.add_artist(legend)
    legend_experiment = ax_abs.legend(
        handles=[off_experiment, on_experiment],
        frameon=False,
        loc="upper right",
        fontsize=7.5,
    )
    legend_red = ax_red.legend(frameon=False, loc="lower left", fontsize=10)
    ax_red.set_xlabel(r"Temperature $T$ ($^\circ$C)")
    ax_red.set_ylabel("Normalized Rayleigh enhancement")
    ax_red.set_xlim(TEMPERATURE_SCAN_C[0], 85.0)
    ax_red.set_ylim(-0.05, 1.19)
    ax_red.text(
        0.50,
        0.85,
        "Resonance:" r"$~\Delta = 0$",
        transform=ax_red.transAxes,
        fontname=plt.rcParams["font.family"],
        fontsize=14,
        ha="left",
        va="top",
    )

    inset = inset_axes(ax_abs, width="30%", height="56%", loc="center right", borderpad=1.5)
    omega_sq = np.asarray(omega_sq_values)
    transition_temperatures = np.asarray(transition_values)
    linear_fit = np.polyfit(omega_sq, transition_temperatures, deg=1)
    omega_sq_line = np.linspace(omega_sq.min(), omega_sq.max(), 200)
    inset.plot(omega_sq, transition_temperatures, "o", color="black", markersize=4.5)
    inset.plot(
        omega_sq_line,
        np.polyval(linear_fit, omega_sq_line),
        color="#c44e52",
        linestyle="--",
        linewidth=1.5,
    )
    inset.set_xlabel(r"$\Omega^2$ (cm$^{-2}$)", fontsize=8)
    inset.set_ylabel(r"$T_R$ ($^\circ$C)", fontsize=8)
    inset.tick_params(labelsize=8)
    ax_abs.text(
        -0.30,
        0.90,
        "(a)",
        transform=ax_abs.transAxes,
        fontname=plt.rcParams["font.family"],
        fontsize=36,
        clip_on=False,
    )
    ax_red.text(
        -0.30,
        0.90,
        "(b)",
        transform=ax_red.transAxes,
        fontname=plt.rcParams["font.family"],
        fontsize=36,
        clip_on=False,
    )
    style_legend(legend)
    style_legend(legend_experiment)
    style_legend(legend_red)
    style_axis_text(ax_abs)
    style_axis_text(ax_red)
    style_axis_text(inset)

    save_figure(fig, "Fig3.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
