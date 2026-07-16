"""Fig 4 — S(q) and h(r)=g(r)-1 comparison: normal liquid, vibrationally polarized, near structural spinodal.

Physical picture
----------------
This figure shows a mesoscopic / long-wavelength OZ comparison using representative
stable-side masses.  All three cases have a_m^eff > 0, so the Gaussian OZ expressions
are controlled.

This figure uses representative stable-side masses and is NOT the same as the
detuning-averaged Rayleigh readout used in Figs. 2 and 3.  The OZ expressions
are controlled for a_m^eff > 0.

The masses are computed from a_m_eff_thermodynamic at DELTA_CM=0, TEMPERATURE_C=20 C:

    Outside cavity (P = 0):
        a_m_eff = a_m = 1.0
        zeta_out  = sqrt(kappa / a_m)

    Vibrationally polarized at OMEGA_R_INTERMEDIATE_CM = 25 cm^{-1}:
        a_m_eff = a_m - g*P^2(Omega=25) ≈ 0.537  (stable side, > 0)
        S(0)/S_out(0) ≈ 1.86,  zeta/zeta_out ≈ 1.36

    Near structural spinodal at OMEGA_R_BRIGHT_CM (computed so a_m^eff = rayleigh_width_a):
        a_m_eff = rayleigh_width_a ≈ 0.0215 > 0  (representative near-threshold, stable-side mass)
        S(0)/S_out(0) ≈ 46.5,  zeta/zeta_out ≈ 6.82

Real-space pair correlations — OZ tail form
-------------------------------------------
    h(r) = g(r) - 1 ≈ A * exp(-r / zeta) / r      [OZ, Yukawa-like tail]

We plot only r >= r_min = 0.3*zeta_out (a short-range cutoff) to avoid the 1/r
singularity at r = 0.  Panel b is normalized by h_out(r_min), where h_out is the
outside-cavity total correlation function.  The short-range liquid structure
(first shell) is not described by this mesoscopic theory and is intentionally omitted.

Assumptions
-----------
- kappa is fixed to 1 in the shared ModelParams object, setting the dimensionless
  length unit for the normalized q and r axes.
- Temperature is set to T = 20 C, Delta = 0.
- The prefactor A is set to A = S(0) / (4*pi*xi^2) so h(r)=g(r)-1 integrates
  consistently with S(0) at the mesoscopic level.
- q is in units of 1/xi_out for clarity.
"""
from __future__ import annotations

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from model import (
    DEFAULT_PARAMS,
    ModelParams,
    a_m_eff_thermodynamic,
    apply_publication_style,
    save_figure,
    style_axis_text,
    style_legend,
)

# ── physical parameters ───────────────────────────────────────────────────────
DELTA_CM = 0.0
TEMPERATURE_C = 20.0
OMEGA_R_INTERMEDIATE_CM = 25.0


def omega_for_target_mass(target_mass: float, params=DEFAULT_PARAMS) -> float:
    """Solve a_m - g*P^2 = target_mass at DELTA_CM and TEMPERATURE_C."""
    p2_target = (params.a_m - target_mass) / params.g
    omega_sq = (
        params.bP * p2_target
        + params.alpha_delta * DELTA_CM**2
        + params.alpha_T * (TEMPERATURE_C - params.T0)
    ) / params.alpha_omega
    return float(np.sqrt(max(omega_sq, 0.0)))


OMEGA_R_BRIGHT_CM = omega_for_target_mass(DEFAULT_PARAMS.rayleigh_width_a)

# ── q and r grids ─────────────────────────────────────────────────────────────
Q_MAX_UNITS = 2.0        # in units of 1/xi_out
N_Q = 500
R_MIN_UNITS = 0.3        # short-range cutoff (in units of xi_out)
R_MAX_UNITS = 10.0
N_R = 600
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
PANEL_LABEL_FONTSIZE = 34


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


def _get_a_m_values(params=DEFAULT_PARAMS) -> tuple[float, float, float]:
    """Return thermodynamic stable-side masses for outside, intermediate, and bright cases."""
    a_out = float(params.a_m)   # P = 0 outside cavity
    a_intermediate = float(a_m_eff_thermodynamic(
        delta_cm=DELTA_CM,
        omega_r_cm=OMEGA_R_INTERMEDIATE_CM,
        temperature_c=TEMPERATURE_C,
        params=params,
    ))
    a_bright = float(a_m_eff_thermodynamic(
        delta_cm=DELTA_CM,
        omega_r_cm=OMEGA_R_BRIGHT_CM,
        temperature_c=TEMPERATURE_C,
        params=params,
    ))
    assert a_intermediate > 0.0, f"a_intermediate = {a_intermediate} is not positive"
    assert a_bright > 0.0, f"a_bright = {a_bright} is not positive"
    return a_out, a_intermediate, a_bright


def _S_OZ(q_xi: np.ndarray, S0: float) -> np.ndarray:
    """OZ structure factor.  q_xi = q * xi (dimensionless)."""
    return S0 / (1.0 + q_xi ** 2)


def _gr_OZ(r: np.ndarray, S0: float, xi: float) -> np.ndarray:
    """OZ real-space tail: h(r)=g(r)-1 = A * exp(-r/xi) / r.

    Prefactor A is set so that
        4*pi * integral_0^inf h(r) r^2 dr = S(0) - 1 ≈ S(0)
    for S(0) >> 1.  For the OZ form A*exp(-r/xi)/r:
        integral = 4*pi*A * xi^2  =>  A = S(0) / (4*pi*xi^2)
    (using rho=1 in dimensionless units).
    """
    A = S0 / (4.0 * np.pi * xi ** 2)
    return A * np.exp(-r / xi) / r


def _panel_Sq(ax: plt.Axes, params: ModelParams) -> None:
    a_out, a_intermediate, a_bright = _get_a_m_values(params)

    xi_out = np.sqrt(params.kappa / a_out)
    xi_intermediate = np.sqrt(params.kappa / a_intermediate)
    xi_bright = np.sqrt(params.kappa / a_bright)
    S0_out = 1.0 / a_out
    S0_intermediate = 1.0 / a_intermediate
    S0_bright = 1.0 / a_bright

    # q in units of 1/xi_out
    q_norm = np.linspace(0.0, Q_MAX_UNITS, N_Q)
    q_xi_out = q_norm
    q_xi_intermediate = q_norm * (xi_intermediate / xi_out)
    q_xi_bright = q_norm * (xi_bright / xi_out)

    Sq_out = _S_OZ(q_xi_out, S0_out)
    Sq_intermediate = _S_OZ(q_xi_intermediate, S0_intermediate)
    Sq_bright = _S_OZ(q_xi_bright, S0_bright)

    # Normalize to S_outside(0) so the ratio is visible
    norm = S0_out
    ax.plot(q_norm, Sq_out / norm, color="0.50", linewidth=2.2,
            linestyle="--", label="Normal Liquid")
    ax.plot(q_norm, Sq_intermediate / norm, color="#8ecae6", linewidth=2.2,
            label=rf"Vibrationally Polarized ($\Omega={OMEGA_R_INTERMEDIATE_CM:.0f}$ cm$^{{-1}}$)")
    ax.plot(q_norm, Sq_bright / norm, color="#1f77b4", linewidth=2.2,
            label=rf"Near structural spinodal ($\Omega={OMEGA_R_BRIGHT_CM:.1f}$ cm$^{{-1}}$)")

    ax.axhline(1.0, color="0.80", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"$q\,\zeta_{\rm out}$ (dimensionless)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$S(q)\,/\,S_{\rm out}(0)$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0.0, Q_MAX_UNITS)
    ax.set_ylim(bottom=-2.0)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    legend = ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=10,
    )
    ax.text(
        -0.24,
        0.90,
        "(a)",
        transform=ax.transAxes,
        fontname=plt.rcParams["font.family"],
        fontsize=PANEL_LABEL_FONTSIZE,
        clip_on=False,
    )
    style_legend(legend)

    inset = inset_axes(ax, width="38%", height="50%", loc="lower right", borderpad=4.3)
    inset.plot(q_norm, Sq_out / norm, color="0.50", linewidth=1.8, linestyle="--")
    inset.plot(q_norm, Sq_intermediate / norm, color="#8ecae6", linewidth=1.8)
    inset.axhline(1.0, color="0.80", linestyle=":", linewidth=0.7)
    inset.set_xlim(0.0, 2.0)
    inset.set_ylim(0.0, 2.0)
    inset.set_xlabel(r"$q\,\zeta_{\rm out}$", fontsize=8)
    inset.tick_params(labelsize=8, direction="in")
    style_axis_text(inset)

    style_axis_text(ax)


def _panel_gr(ax: plt.Axes, params: ModelParams) -> None:
    a_out, a_intermediate, a_bright = _get_a_m_values(params)

    xi_out = np.sqrt(params.kappa / a_out)
    xi_intermediate = np.sqrt(params.kappa / a_intermediate)
    xi_bright = np.sqrt(params.kappa / a_bright)
    S0_out = 1.0 / a_out
    S0_intermediate = 1.0 / a_intermediate
    S0_bright = 1.0 / a_bright

    # r in units of xi_out
    r_norm = np.linspace(R_MIN_UNITS, R_MAX_UNITS, N_R)
    r = r_norm * xi_out

    gr_out = _gr_OZ(r, S0_out, xi_out)
    gr_intermediate = _gr_OZ(r, S0_intermediate, xi_intermediate)
    gr_bright = _gr_OZ(r, S0_bright, xi_bright)

    # Normalize to peak of outside curve
    norm = float(gr_out.max()) if gr_out.max() > 0 else 1.0
    ax.plot(r_norm, gr_out / norm, color="0.50", linewidth=2.2,
            linestyle="--", label="Normal Liquid")
    ax.plot(r_norm, gr_intermediate / norm, color="#8ecae6", linewidth=2.2,
            label=rf"Vibrationally Polarized ($\Omega={OMEGA_R_INTERMEDIATE_CM:.0f}$ cm$^{{-1}}$)")
    ax.plot(r_norm, gr_bright / norm, color="#1f77b4", linewidth=2.2,
            label=rf"Near structural spinodal ($\Omega={OMEGA_R_BRIGHT_CM:.1f}$ cm$^{{-1}}$)")

    ax.axhline(0.0, color="0.85", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$r\,/\,\zeta_{\rm out}$ (dimensionless)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        r"$h(r)\,/\,h_{\rm out}(r_{\rm min})$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_xlim(R_MIN_UNITS, R_MAX_UNITS)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    # legend = ax.legend(frameon=False, loc="upper right", fontsize=11)
    ax.text(
        -0.27,
        0.90,
        "(b)",
        transform=ax.transAxes,
        fontname=plt.rcParams["font.family"],
        fontsize=PANEL_LABEL_FONTSIZE,
        clip_on=False,
    )
    # style_legend(legend)

    # Mark correlation lengths
    ax.axvline(1.0, color="0.50", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(xi_intermediate / xi_out, color="#8ecae6", linestyle="--", linewidth=1.0)
    ax.axvline(xi_bright / xi_out, color="#1f77b4", linestyle="--", linewidth=1.0)
    ax.text(1.2, 0.56, r"$\zeta_{\rm out}$", fontsize=16, color="0.50",
            transform=ax.get_xaxis_transform())
    ax.text(1.55, 0.40,
            rf"$\zeta_{{\rm in}}(\Omega={OMEGA_R_INTERMEDIATE_CM:.0f}\,\mathrm{{cm}}^{{-1}})$"
            #"\n"
            rf"$= {xi_intermediate/xi_out:.2f}\,\zeta_{{\rm out}}$",
            fontsize=10, color="#4a9fc2",
            transform=ax.get_xaxis_transform())
    ax.text(1.55, 0.22,
            rf"$\zeta_{{\rm in}}(\Omega={OMEGA_R_BRIGHT_CM:.1f}\,\mathrm{{cm}}^{{-1}})$"
            #"\n"
            rf"$= {xi_bright/xi_out:.2f}\,\zeta_{{\rm out}}$",
            fontsize=10, color="#1f77b4",
            transform=ax.get_xaxis_transform())
    style_axis_text(ax)


def main() -> None:
    apply_publication_style()
    params = DEFAULT_PARAMS

    a_out, a_intermediate, a_bright = _get_a_m_values(params)
    xi_out = np.sqrt(params.kappa / a_out)
    xi_intermediate = np.sqrt(params.kappa / a_intermediate)
    xi_bright = np.sqrt(params.kappa / a_bright)
    sys.stdout.buffer.write(
        (f"OMEGA_R_BRIGHT_CM            = {OMEGA_R_BRIGHT_CM:.4f} cm^-1\n"
         f"a_m_eff  outside:            {a_out:.4f}\n"
         f"         intermediate:       {a_intermediate:.6f}\n"
         f"         bright:             {a_bright:.6f}\n"
         f"xi/xi_out  outside:          {xi_out/xi_out:.4f}\n"
         f"           intermediate:     {xi_intermediate/xi_out:.4f}\n"
         f"           bright:           {xi_bright/xi_out:.4f}\n"
         f"S(0)/S_out(0)  intermediate: {a_out/a_intermediate:.4f}\n"
         f"               bright:       {a_out/a_bright:.4f}\n").encode()
    )

    fig, (ax_Sq, ax_gr) = plt.subplots(
        1, 2,
        figsize=(9.5, 4.0),
        gridspec_kw={"wspace": 0.28},
    )

    _panel_Sq(ax_Sq, params)
    _panel_gr(ax_gr, params)

    save_figure(fig, "Fig4.pdf")
    plt.close(fig)
    sys.stdout.buffer.write(b"Saved: Fig4.pdf\n")


if __name__ == "__main__":
    main()
