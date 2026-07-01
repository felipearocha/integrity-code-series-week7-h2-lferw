# INTEGRITY CODE SERIES — Week 7
## Hydrogen Conversion of Aging LF-ERW Pipeline: Coupled Diffusion-Fracture Life Prediction

[![CI](https://github.com/felipearocha/integrity_code_series_week7_h2_lferw/actions/workflows/ci.yml/badge.svg)](https://github.com/felipearocha/integrity_code_series_week7_h2_lferw/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 182 passing](https://img.shields.io/badge/tests-182%20passing-brightgreen.svg)](tests)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172481.svg)](https://doi.org/10.5281/zenodo.20172481)

---

## Problem Statement

Can a 1959-vintage 20-inch API 5L X52 low-frequency ERW pipeline with pre-existing selective seam corrosion pits be safely converted from natural gas to hydrogen service?

This is the same pipe specification that failed at Willow River, Minnesota on January 16, 2026 (PHMSA Corrective Action Order).

## News Hooks (March 2026)

- AMPP Pipeline Industry Report 2026: corrosion incidents rose to 25%+ of all pipeline failures
- PHMSA Corrective Action Order on Willow River LF-ERW explosion
- ASME absorbing B31.12 hydrogen requirements into B31.8 (2026 edition)
- 3,072 miles of pre-1970 pipe in Minnesota alone, 209 miles of unknown vintage

## Governing Equations

[**view the full rendered reference**](https://htmlpreview.github.io/?https://github.com/felipearocha/integrity_code_series_week7_h2_lferw/blob/main/docs/equations.html)

Every constant is tagged `[ASSUMED]` (requires experimental calibration) or is a
published standard / physical-constant value. The physics is a 5-mechanism
sequential chain; the headline equations are reproduced below (GitHub renders the
math natively).

**1. Hydrogen diffusion — Oriani stress-assisted Fick's law** (`src/hydrogen_diffusion.py`):

$$ \frac{\partial C_L}{\partial t} \;=\; D_L\,\frac{\partial^2 C_L}{\partial x^2} \;+\; D_L\,\frac{V_H}{RT}\,\frac{\partial}{\partial x}\!\left(C_L\,\frac{\partial \sigma_h}{\partial x}\right) $$

with Sievert's-law inner-surface boundary condition:

$$ C_L(0,t) = C_0 = S\sqrt{p_{H_2}}, \qquad C_L(w,t) = 0, \qquad C_L(x,0) = 0 $$

**2. Pit growth at the ERW seam** (power law from prior NG service, `src/pit_to_crack.py`):

$$ a_{\text{pit}}(t) \;=\; k_{\text{pit}}\,f_{\text{seam}}\,t^{\,n_{\text{pit}}} $$

**3. Pit-to-crack transition** (Murakami $\sqrt{\text{area}}$ SIF vs hydrogen-degraded threshold):

$$ K_{\text{pit}} \;=\; 0.65\,\sigma\,\sqrt{\pi\sqrt{\text{area}}}, \qquad \text{area} = \frac{\pi}{2}\,c\,a, \qquad c = (c/a)\,a $$

$$ K_{th}(C_H) \;=\; K_{th,\text{air}}\,\exp\!\left(-\lambda_{th}\,\frac{C_H}{C_{\text{ref}}}\right), \qquad K_{th}(C_H) \ge K_{th,\min} $$

**4. Hydrogen-assisted fatigue crack growth** (modified Paris law, `src/ha_fcg.py`):

$$ \frac{da}{dN} \;=\; C_{\text{paris}}\,(\Delta K)^{m}\,\bigl[\,1 + \alpha_H\,(C_H/C_{\text{ref}})^{\beta_H}\,\bigr] $$

**5. Failure criterion** (hydrogen-degraded toughness):

$$ K_{\max} \;\ge\; K_{IC}(C_H), \qquad K_{IC}(C_H) = K_{IC,\text{air}}\,\exp\!\left(-\lambda_K\,\frac{C_H}{C_{\text{ref}}}\right), \qquad K_{IC}(C_H) \ge K_{IC,\min} $$

**MAOP under ASME B31.12** (`src/ha_fcg.py → maop_under_b3112`):

$$ P \;=\; \frac{2\,\text{SMYS}\,t\,F\,H_f}{D} $$

Full rendered (MathJax) reference with sources and code cross-references:
**[docs/equations.html](docs/equations.html)** — open in any browser.

## Repository Structure

```
integrity_code_series_week7_h2_lferw/
    src/
        config.py                    # All parameters, [ASSUMED] flags
        hydrogen_diffusion.py        # 1D through-wall H diffusion PDE
        pit_to_crack.py              # Murakami SIF, transition criteria
        ha_fcg.py                    # HA-FCG + integrated life engine
        monte_carlo.py               # MC + LHS parametric sweep
        ml_surrogate.py              # GBR surrogate, feature ranking
        cybersecurity.py             # STRIDE, audit chain, sensor integrity
        visualization.py             # All plots + animated GIF
    tests/
        test_suite.py                # 182 tests
    docs/
        equations.html               # Rendered (MathJax) governing equations
    assets/                          # Generated visuals + surrogate model
    README.md
    requirements.txt
```

## Execution Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run test suite
cd tests
python -m pytest test_suite.py -v

# 3. Run full simulation (from src/)
cd ../src
python -c "
from config import default_config
from hydrogen_diffusion import solve_diffusion
from ha_fcg import run_life_prediction, cycles_to_years
from monte_carlo import run_monte_carlo, run_lhs_sweep
from ml_surrogate import train_surrogate, validate_feature_ranking
from visualization import *

cfg = default_config()
print('Running hydrogen diffusion...')
diff = solve_diffusion(cfg['geom'], cfg['h2_transport'], cfg['ops'], cfg['sim'])
print(f'Diffusion solved: {len(diff.t_grid)} snapshots')

print('Running life prediction...')
life = run_life_prediction(
    cfg['geom'], cfg['mat'], cfg['h2_transport'],
    cfg['h2_embrittlement'], cfg['pit'], cfg['ops'], cfg['sim'],
    diffusion_result=diff
)
print(f'Failure at cycle {life.failure_cycle} ({cycles_to_years(life.failure_cycle, cfg[\"ops\"]):.1f} years)')
print(f'Failure mode: {life.failure_mode}')

print('Running Monte Carlo (10,000 samples)...')
mc = run_monte_carlo(
    cfg['geom'], cfg['mat'], cfg['h2_transport'],
    cfg['h2_embrittlement'], cfg['pit'], cfg['ops'], cfg['sim']
)
print(f'P5={mc.percentile_5:.1f} yr, P50={mc.percentile_50:.1f} yr, P95={mc.percentile_95:.1f} yr')
print(f'Fraction below 10yr: {mc.fraction_below_10yr:.1%}')

print('Running LHS sweep (2,000 samples)...')
lhs = run_lhs_sweep(
    cfg['geom'], cfg['mat'], cfg['h2_transport'],
    cfg['h2_embrittlement'], cfg['pit'], cfg['ops'], cfg['sim']
)
print('Spearman correlations:')
for k, v in sorted(lhs.spearman_correlations.items(), key=lambda x: -abs(x[1])):
    print(f'  {k}: {v:.3f}')

print('Training GBR surrogate...')
surr = train_surrogate(lhs.params, lhs.remaining_life_years, lhs.param_names)
print(f'R2 test: {surr.r2_test:.3f}')
ranking = validate_feature_ranking(surr.feature_importances)
for k, v in ranking.items():
    print(f'  {k}: {v}')

print('Generating visuals...')
plot_hero_diffusion_crack(diff, life, cfg['geom'])
plot_monte_carlo_cdf(mc)
plot_sensitivity_tornado(lhs.spearman_correlations)
plot_surrogate_parity(surr)
generate_gif(diff, life, cfg['geom'])
print('All outputs generated in assets/')
"
```

## Key Assumptions Requiring Experimental Calibration

All parameters flagged [ASSUMED] in config.py. Critical ones:

- Sievert's constant S: governs surface hydrogen uptake
- Hydrogen embrittlement exponents (lambda_K, lambda_th): control property degradation rate
- HA-FCG enhancement parameters (alpha_H, beta_H): control crack growth acceleration
- ERW seam toughness K_IC and threshold K_th: must be measured on actual seam material in pressurized H2
- Seam enhancement factor f_seam: depends on specific manufacturing vintage

## Escalation Table

| Dimension | Week 5 (MIC) | Week 6 (Galvanic) | **Week 7 (H2+LF-ERW)** |
|---|---|---|---|
| Physics coupling | Reaction-diffusion ODE | 2D Laplace + BV | **Transport PDE + fracture ODE** |
| Mechanism chain | Single (biofilm + acid) | Single (galvanic) | **5-mechanism sequential** |
| Probabilistic | Parametric sweep only | MCMC inverse | **Monte Carlo 10,000 + LHS** |
| Spatial | 1D chainage | 2D cross-section | **1D radial + through-wall** |
| Regulatory | None | None | **ASME B31.12 / PHMSA mapping** |

## Cybersecurity (STRIDE)

7 threats documented (STRIDE + data poisoning). SHA-256 hash-chain audit logging.
Sensor spoofing detection for ILI data. Physics monotonicity check for training data
integrity. See `src/cybersecurity.py`.

## Anti-Hallucination Note

Every parameter in `config.py` is either a published standard / physical-constant value
or explicitly flagged `[ASSUMED]`. The tags are applied honestly across three tiers:

- **T1 — standard / handbook / measured:** physical constants (Faraday, gas constant),
  API 5L X52 specification values (SMYS 358 MPa, SMTS 455 MPa), the ASME B31.12 design
  factor 0.50 (Table PL-3.7.1), and the partial molar volume of H in Fe (literature
  consensus).
- **T2 — established closed-form derivations:** the Lame thick-cylinder stress field,
  Barlow hoop stress, the Murakami √area surface-defect SIF, the El Haddad short-crack
  size, the simplified Newman-Raju surface-crack SIF, and the Paris-law form.
- **T3 — `[ASSUMED]` phenomenological fits requiring calibration:** the Sievert's
  constant S, lattice diffusivity D_L, the hydrogen-degradation exponents (lambda_K,
  lambda_th), the HA-FCG enhancement parameters (alpha_H, beta_H), the ERW-seam
  toughness/threshold values, the pit-growth constants (k_pit, n_pit, f_seam), and the
  B31.12 material performance factor H_f.

No equation, constant, or citation in this repository is invented: T3 values are marked
`[ASSUMED]` at the point of use and must be measured in pressurized hydrogen before any
engineering application.

## Disclaimer

Research tool only. Not for design, fitness-for-service, or safety-critical decisions
without site-specific calibration and independent PE review.

Not for production engineering decisions without experimental calibration of all
`[ASSUMED]` parameters. The hydrogen-degradation and HA-FCG enhancement functions are
phenomenological fits, not measured constants.

## License

MIT — Felipe Rocha. See [LICENSE](LICENSE).

## How to Cite

If this software contributes to your work, please cite both the software (this repository) and the underlying methods it implements.

**Software (archived release):**

> Rocha, F. (2026). *Integrity Code Series - Week 7 - Hydrogen Conversion of Aging LF-ERW Pipeline: Coupled Diffusion-Fracture Life Prediction* (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20172481

**BibTeX:**

```bibtex
@software{rocha_2026_week7,
  author       = {Rocha, Felipe},
  title        = {{Integrity Code Series - Week 7 - Hydrogen Conversion of Aging LF-ERW Pipeline: Coupled Diffusion-Fracture Life Prediction}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.20172481},
  url          = {https://doi.org/10.5281/zenodo.20172481}
}
```

| DOI | Points to |
|-----|-----------|
| [`10.5281/zenodo.20172481`](https://doi.org/10.5281/zenodo.20172481) (concept) | Always resolves to the latest version — use this for citation. |
| [`10.5281/zenodo.20172482`](https://doi.org/10.5281/zenodo.20172482) (version) | Pinned to v1.0.1 specifically — use when reproducibility matters. |

A machine-readable citation file is also available in [`CITATION.cff`](CITATION.cff) - GitHub will display a "Cite this repository" widget at the top right of the repo page that exports BibTeX / APA / RIS automatically.

## Integrity Code Series

Part of an ongoing series of physics-first integrity simulators by Felipe Rocha:

| # | Repo | Domain |
|---|---|---|
| Week 3 | [Integrity-code-series-3](https://github.com/felipearocha/Integrity-code-series-3) | F1 lap simulation (six coupled ODEs) |
| Week 6 | [integrity-code-series-week6-smartphone-galvanic](https://github.com/felipearocha/Integrity-code-series-week6-smartphone-galvanic) | Smartphone galvanic corrosion (Laplace + Butler-Volmer) |
| **Week 7** | **[integrity_code_series_week7_h2_lferw](https://github.com/felipearocha/integrity_code_series_week7_h2_lferw)** | **LF-ERW H2 conversion (B31.12 + NACE TM0316) — this repo** |
| Week 8 | [integrity-code-series-week8-creep-fatigue-heater](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater) | Creep-fatigue 9Cr-1Mo (Norton/Omega + Coffin-Manson) |
| Week 9 | [integrity-code-series-week9-cui](https://github.com/felipearocha/integrity-code-series-week9-cui) | CUI thermohygro-electrochemical (3 PDEs, Strang) |
| Week 10 | [integrity-code-series-week-10_nnph_scc](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc) | NNpHSCC full-physics (Chen-Sutherby-Xing + BS 7910) |
| Week 11 | [integrity-code-series-week11-erosion-corrosion-multiphase](https://github.com/felipearocha/integrity-code-series-week11-erosion-corrosion-multiphase) | Erosion-corrosion multiphase (NORSOK M-506 + DNV-RP-O501 + G119 + API 579) |
