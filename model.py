from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.legend import Legend


@dataclass(frozen=True)
class ModelParams:
    """Dimensionless GL parameters exposed with experiment-like units."""

    T0: float = 5.0
    alpha_T: float = 3.43e-2
    alpha_delta: float = 1.0e-2
    alpha_omega: float = 4.0e-3
    a_m: float = 1.0
    g: float = 0.233
    bP: float = 1.0
    b_m: float = 1.0
    kappa: float = 1.0
    rayleigh_width_a: float = 0.0215


DEFAULT_PARAMS = ModelParams()


def _resolve_font_family(
    preferred_names: tuple[str, ...],
    fallback_family: str,
) -> str:
    """Return the best available font family matching the preferred names."""
    direct_match = {
        entry.name.lower(): entry.name for entry in fm.fontManager.ttflist
    }
    for name in preferred_names:
        matched = direct_match.get(name.lower())
        if matched is not None:
            return matched

    terms = tuple(name.lower() for name in preferred_names)
    candidates: list[Path] = []
    seen_paths: set[Path] = set()

    for entry in fm.fontManager.ttflist:
        path = Path(entry.fname)
        if path in seen_paths:
            continue
        name = entry.name.lower()
        if any(term in name for term in terms):
            candidates.append(path)
            seen_paths.add(path)

    for font_path in fm.findSystemFonts():
        path = Path(font_path)
        if path in seen_paths:
            continue
        if any(term in path.name.lower() for term in terms):
            candidates.append(path)
            seen_paths.add(path)

    for font_dir in (Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts")):
        if not font_dir.exists():
            continue
        for path in font_dir.iterdir():
            if path in seen_paths or not path.is_file():
                continue
            if path.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
                continue
            if any(term in path.name.lower() for term in terms):
                candidates.append(path)
                seen_paths.add(path)

    if not candidates:
        return fallback_family

    def candidate_score(path: Path) -> tuple[int, int]:
        filename = path.name.lower()
        score = 0
        if "regular" in filename or "roman" in filename:
            score -= 2
        if "bold" in filename or "black" in filename:
            score += 2
        if "italic" in filename or "oblique" in filename:
            score += 1
        return score, len(filename)

    best_path = sorted(candidates, key=candidate_score)[0]
    fm.fontManager.addfont(str(best_path))
    return fm.FontProperties(fname=str(best_path)).get_name()


HELVETICA_FAMILY = _resolve_font_family(
    ("Helvetica", "Arial"),
    fallback_family="DejaVu Sans",
)
def a_P(
    delta_cm: np.ndarray | float,
    omega_r_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    delta = np.asarray(delta_cm, dtype=float)
    omega_r = np.asarray(omega_r_cm, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float)
    return (
        params.alpha_T * (temperature - params.T0)
        + params.alpha_delta * delta**2
        - params.alpha_omega * omega_r**2
    )


def P_squared(
    delta_cm: np.ndarray | float,
    omega_r_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    driving = -a_P(delta_cm, omega_r_cm, temperature_c, params=params)
    return np.maximum(driving, 0.0) / params.bP


def P(
    delta_cm: np.ndarray | float,
    omega_r_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    return np.sqrt(P_squared(delta_cm, omega_r_cm, temperature_c, params))


def structural_order(
    delta_cm: np.ndarray | float,
    omega_r_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    # With coupling -g m^2 P^2 / 2, the structural mode has Hessian a_m - g*P^2 at m=0.
    # The mean field is m=0 when stable (g*P^2 < a_m), and sqrt((g*P^2 - a_m)/b_m) when ordered.
    gP2 = params.g * P_squared(delta_cm, omega_r_cm, temperature_c, params)
    return np.sqrt(np.maximum((gP2 - params.a_m) / params.b_m, 0.0))


def a_m_eff_thermodynamic(
    delta_cm: np.ndarray | float,
    omega_r_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Raw thermodynamic structural mass a_m^eff = a_m - g*P^2. No floor applied."""
    return params.a_m - params.g * P_squared(delta_cm, omega_r_cm, temperature_c, params)


def structure_factor_proxy(
    delta_cm: np.ndarray | float,
    omega_r_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    return 1.0 / a_m_eff_thermodynamic(delta_cm, omega_r_cm, temperature_c, params)

def rayleigh_activation(
    delta_cm: np.ndarray | float,
    omega_r_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Rayleigh readout functional S_R(a_m^eff) = [1 + exp(a_m^eff/sigma_a)]^-1.

    A phenomenological measurement functional of the thermodynamic mass; it
    does not modify the GL free energy and is not itself a phase boundary.
    """
    a_eff = a_m_eff_thermodynamic(delta_cm, omega_r_cm, temperature_c, params)
    scaled = np.clip(a_eff / params.rayleigh_width_a, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(scaled))


def rayleigh_enhancement_ratio(
    delta_cm: np.ndarray | float,
    omega_r_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Dimensionless plotting signal 1 + S_R(a_m^eff).

    The absolute amplitude I0 is omitted because the figure data use normalized
    Rayleigh excess signals, where I0 cancels.
    """
    activation = rayleigh_activation(delta_cm, omega_r_cm, temperature_c, params)
    return 1.0 + activation

def critical_rabi_splitting(
    delta_cm: np.ndarray | float,
    temperature_c: np.ndarray | float,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    delta = np.asarray(delta_cm, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float)
    omega_sq = (
        params.alpha_delta / params.alpha_omega * delta**2
        + (params.alpha_T / params.alpha_omega) * (temperature - params.T0)
    )
    return np.sqrt(np.maximum(omega_sq, 0.0))


def critical_temperature(
    omega_r_cm: np.ndarray | float,
    delta_cm: np.ndarray | float = 0.0,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    omega_r = np.asarray(omega_r_cm, dtype=float)
    delta = np.asarray(delta_cm, dtype=float)
    return params.T0 + (
        params.alpha_omega * omega_r**2 - params.alpha_delta * delta**2
    ) / params.alpha_T


def rayleigh_transition_temperature(
    omega_r_cm: np.ndarray | float,
    delta_cm: np.ndarray | float = 0.0,
    params: ModelParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Structural spinodal temperature defined by a_m^eff = 0, i.e. P^2 = a_m/g."""
    omega_r = np.asarray(omega_r_cm, dtype=float)
    delta = np.asarray(delta_cm, dtype=float)
    return params.T0 + (
        params.alpha_omega * omega_r**2
        - params.alpha_delta * delta**2
        - params.bP * params.a_m / params.g
    ) / params.alpha_T

RABI_SPLITTING_AT_1M_CM: float = 83.0
"""Collective Rabi splitting (cm^-1) reported by Patrahau et al. for a 1.0 M solution."""


def concentration_to_omega_sq(
    concentration_m: np.ndarray | float,
    rabi_splitting_at_1m_cm: float = RABI_SPLITTING_AT_1M_CM,
) -> np.ndarray:
    """Omega^2 = C * (Rabi splitting at 1 M)^2, since Omega is proportional to sqrt(C)."""
    concentration = np.asarray(concentration_m, dtype=float)
    return concentration * rabi_splitting_at_1m_cm**2


EQUILIBRIUM_A_M: float = 1.0
"""Bare structural mass a_m for the NMR equilibrium calibration, fixed to 1 by the same
field-normalization convention as a_m in Table 1 of the main text (see
apply_publication_style docstring context and Sec.~"Model parameters" of the main text)."""

EQUILIBRIUM_ALPHA_OMEGA: float = 1.0
"""Omega^2-to-a_P coupling for the NMR calibration, fixed to 1 by convention.

The on-resonance series holds Delta=0 and a single fixed experimental temperature, so
alpha_Delta, alpha_T, and T0 (Eq.~aP of the main text) are not probed and g cannot be
separated from alpha_Omega using this dataset alone (only the product g*alpha_Omega/b_P
is identifiable). Fixing alpha_Omega=1 and b_P=1 (Table 1 convention) absorbs this
product entirely into g, which becomes the sole free GL-sector fit parameter below.
"""


def equilibrium_structural_mass_omega_sq(
    omega_sq: np.ndarray | float,
    g: float,
    a_m: float = EQUILIBRIUM_A_M,
    alpha_omega: float = EQUILIBRIUM_ALPHA_OMEGA,
) -> np.ndarray:
    """a_m^eff(Omega^2) = a_m - g*alpha_omega*Omega^2 (Eq.~aeff of the main text at
    fixed Delta=0 and fixed T, i.e. b_P=1 and the alpha_Delta, alpha_T terms dropped).
    """
    omega_sq_arr = np.asarray(omega_sq, dtype=float)
    return a_m - g * alpha_omega * omega_sq_arr


def equilibrium_activation(
    a_m_eff: np.ndarray | float,
    sigma_eq: float,
) -> np.ndarray:
    """S_eq(a_m^eff) = [1 + exp(a_m^eff/sigma_eq)]^-1.

    Identical in form to S_R of Eq.~sigmoid of the main text (Eq.~SI_Seq_readout).
    """
    a_eff = np.asarray(a_m_eff, dtype=float)
    scaled = np.clip(a_eff / sigma_eq, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(scaled))


def equilibrium_readout(
    a_m_eff: np.ndarray | float,
    K_u: float,
    delta_K: float,
    sigma_eq: float,
) -> np.ndarray:
    """K(a_m^eff) = K_u - delta_K * S_eq(a_m^eff).

    Identical in form to Eq.~readout of the main text (Eq.~SI_K_eq_readout).
    """
    return K_u - delta_K * equilibrium_activation(a_m_eff, sigma_eq)


def equilibrium_readout_omega_sq(
    omega_sq: np.ndarray | float,
    K_u: float,
    delta_K: float,
    g: float,
    sigma_eq: float,
) -> np.ndarray:
    """K(Omega^2) = equilibrium_readout(equilibrium_structural_mass_omega_sq(...)).

    Composition used for curve fitting: the readout is a function of a_m^eff exactly
    as in Eqs.~readout-sigmoid of the main text, with a_m^eff itself evaluated along
    the Omega^2 axis (fixed Delta=0, fixed T) instead of the temperature axis of
    Fig.~3b.
    """
    a_eff = equilibrium_structural_mass_omega_sq(omega_sq, g)
    return equilibrium_readout(a_eff, K_u, delta_K, sigma_eq)


def equilibrium_structural_spinodal_omega_sq(
    g: float,
    a_m: float = EQUILIBRIUM_A_M,
    alpha_omega: float = EQUILIBRIUM_ALPHA_OMEGA,
) -> float:
    """Omega^2 at which a_m^eff=0 (structural spinodal), a derived quantity, not a
    separately fitted parameter."""
    return a_m / (g * alpha_omega)


def load_nmr_equilibrium_series(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a digitized [Concentration (M), K_upper, K_value, K_lower] series.

    Returns (omega_sq, K_center, K_err_lower, K_err_upper) with concentration
    converted to Omega^2 via concentration_to_omega_sq.
    """
    data = np.loadtxt(path)
    concentration_m = data[:, 0]
    k_upper = data[:, 1]
    k_center = data[:, 2]
    k_lower = data[:, 3]
    omega_sq = concentration_to_omega_sq(concentration_m)
    return omega_sq, k_center, k_center - k_lower, k_upper - k_center


@dataclass(frozen=True)
class EquilibriumFitResult:
    """Best-fit parameters (and 1-sigma errors) for equilibrium_readout_omega_sq.

    g is the only fitted GL-sector parameter (a_m=alpha_Omega=b_P=1 by convention, see
    EQUILIBRIUM_A_M/EQUILIBRIUM_ALPHA_OMEGA); K_u, delta_K, sigma_eq are readout-only
    parameters, exactly as sigma_a and I0 are readout-only in Table 2 of the main text.
    """

    K_u: float
    delta_K: float
    g: float
    sigma_eq: float
    K_u_err: float
    delta_K_err: float
    g_err: float
    sigma_eq_err: float

    @property
    def omega_sq_sp(self) -> float:
        """Derived structural spinodal Omega^2 (a_m^eff=0), not a fitted parameter."""
        return equilibrium_structural_spinodal_omega_sq(self.g)

    @property
    def omega_sq_sp_err(self) -> float:
        """Propagated 1-sigma error on omega_sq_sp from g_err (delta method)."""
        return self.omega_sq_sp * (self.g_err / self.g)


MIN_SIGMA_EQ: float = 0.05
"""Lower bound on the fitted readout width sigma_eq (dimensionless, a_m=1 units).

The on-resonance series has only a single concentration bin (0.40-0.50 M) spanning
the drop in K, so an unconstrained fit collapses sigma_eq toward zero, mimicking an
ideal step function with artificially tight formal uncertainties on g and sigma_eq.
This floor (comparable in scale to sigma_a=0.0215 in Table 2 of the main text) keeps
the fitted sigmoid visibly smooth, consistent with the finite-resolution readout
picture of Eq.~SI_Seq_readout, while only mildly degrading the fit quality (chi^2
rises from 1.48 to 1.92 over 3 degrees of freedom).
"""


def fit_equilibrium_readout(
    omega_sq: np.ndarray,
    K_center: np.ndarray,
    K_sigma: np.ndarray | None = None,
    min_sigma_eq: float = MIN_SIGMA_EQ,
) -> EquilibriumFitResult:
    """Fit equilibrium_readout_omega_sq to an on-resonance NMR series via curve_fit."""
    from scipy.optimize import curve_fit

    omega_sq = np.asarray(omega_sq, dtype=float)
    K_center = np.asarray(K_center, dtype=float)

    p0 = (
        float(np.max(K_center)),
        float(np.max(K_center) - np.min(K_center)),
        1.0 / float(np.median(omega_sq)),
        float(min_sigma_eq * 1.5),
    )
    bounds = ((0.0, 0.0, 0.0, min_sigma_eq), (np.inf, np.inf, np.inf, np.inf))

    popt, pcov = curve_fit(
        equilibrium_readout_omega_sq,
        omega_sq,
        K_center,
        p0=p0,
        sigma=K_sigma,
        absolute_sigma=K_sigma is not None,
        bounds=bounds,
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    return EquilibriumFitResult(
        K_u=float(popt[0]),
        delta_K=float(popt[1]),
        g=float(popt[2]),
        sigma_eq=float(popt[3]),
        K_u_err=float(perr[0]),
        delta_K_err=float(perr[1]),
        g_err=float(perr[2]),
        sigma_eq_err=float(perr[3]),
    )


def output_dir() -> Path:
    path = Path(__file__).resolve().parent / "Figures"
    path.mkdir(exist_ok=True)
    return path


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = output_dir() / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def style_legend(legend: Legend | None) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontname(HELVETICA_FAMILY)
        text.set_fontweight("normal")
        text.set_fontstyle("normal")
    title = legend.get_title()
    if title is not None:
        title.set_fontname(HELVETICA_FAMILY)
        title.set_fontweight("normal")
        title.set_fontstyle("normal")


def style_axis_text(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontname(HELVETICA_FAMILY)
    ax.xaxis.label.set_fontweight("normal")
    ax.xaxis.label.set_fontstyle("normal")
    ax.yaxis.label.set_fontname(HELVETICA_FAMILY)
    ax.yaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontstyle("normal")
    ax.title.set_fontname(HELVETICA_FAMILY)
    ax.title.set_fontweight("normal")
    ax.title.set_fontstyle("normal")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(HELVETICA_FAMILY)
        label.set_fontweight("normal")
        label.set_fontstyle("normal")
    for text in ax.texts:
        text.set_fontname(HELVETICA_FAMILY)
        text.set_fontweight("normal")
        text.set_fontstyle("normal")


def _sanity_check(params: ModelParams = DEFAULT_PARAMS) -> None:
    g = params.g
    p_spinodal_sq = params.a_m / params.g
    bmbP_minus_g2 = params.b_m * params.bP - params.g ** 2
    print(f"g                    = {g:.10f}")
    print(f"P_spinodal^2 (a_m^eff=0) = {p_spinodal_sq:.6f}")
    print(f"rayleigh_width_a     = {params.rayleigh_width_a}")
    print(f"b_m * b_P - g^2      = {bmbP_minus_g2:.6f}  (must be > 0)")


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 4.2),
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "axes.labelweight": "normal",
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "font.family": HELVETICA_FAMILY,
            "font.sans-serif": [HELVETICA_FAMILY, "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "font.style": "normal",
            "font.weight": "normal",
            "legend.fontsize": 9,
            "lines.linewidth": 2.2,
            "mathtext.fontset": "custom",
            "mathtext.default": "regular",
            "mathtext.rm": HELVETICA_FAMILY,
            "mathtext.it": f"{HELVETICA_FAMILY}:italic",
            "mathtext.bf": f"{HELVETICA_FAMILY}:bold",
            "mathtext.sf": HELVETICA_FAMILY,
            "mathtext.cal": HELVETICA_FAMILY,
            "mathtext.tt": HELVETICA_FAMILY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "xtick.direction": "in",
            "xtick.labelsize": 10,
            "ytick.direction": "in",
            "ytick.labelsize": 10,
        }
    )


if __name__ == "__main__":
    _sanity_check()
