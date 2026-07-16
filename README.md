# GL Phenomenological Theory — Figure Generation

Python scripts and outputs for the figures of my manuscript 
"A Mesoscopic Ginzburg--Landau Model for Vibrational Strong Coupling Enhanced Rayleigh Scattering in Molecular Liquids" (to be appeared on arXiv).
First released on 07/16/2026.

## Contents

- `model.py` — shared model definitions (`ModelParams`, default parameters) and
  plotting utilities (publication styling, font resolution, figure saving) used
  by all `fig*.py` scripts.
- `fig2.py` — Fig. 2, phase boundary (detuning–Rabi splitting diagram).
- `fig3.py` — Fig. 3, direct Rayleigh readout (normalized excess intensities,
  main-text Eqs. 5–6).
- `fig4.py` — Fig. 4, S(q) and h(r) = g(r) − 1 comparison across normal liquid,
  vibrationally polarized, and near-structural-spinodal regimes.
- `figs1.py` — Fig. S1, detuning–temperature phase diagram at fixed Rabi splittings.
- `figs2.py` — Fig. S2, robustness of the thermal-collapse readout to the
  sigmoid width σ_a (repeats the Fig. 3b scan).
- `Fig1/` — standalone Fig. 1 assets (free-energy schematic, molecular-cluster
  cartoons) and their generating scripts.
- `Figures/` — rendered PDF outputs for the main text and SI figures.
- `TOC graphic/` — table-of-contents graphic assets and generating scripts.

## Usage

Each `fig*.py` script is self-contained aside from importing shared helpers from
`model.py`, and writes its output PDF into `Figures/`. Run with:

```bash
python fig2.py
```

Requires `numpy` and `matplotlib`.
