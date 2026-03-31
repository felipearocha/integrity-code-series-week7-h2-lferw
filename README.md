# INTEGRITY CODE SERIES -- Week 7
## Hydrogen Conversion of Aging LF-ERW Pipeline: Coupled Diffusion-Fracture Life Prediction

### Problem Statement

Can a 1959-vintage 20-inch API 5L X52 low-frequency ERW pipeline with pre-existing selective seam corrosion pits be safely converted from natural gas to hydrogen service?

This is the same pipe specification that failed at Willow River, Minnesota on January 16, 2026 (PHMSA Corrective Action Order).

### News Hooks (March 2026)

- AMPP Pipeline Industry Report 2026: corrosion incidents rose to 25%+ of all pipeline failures
- PHMSA Corrective Action Order on Willow River LF-ERW explosion
- ASME absorbing B31.12 hydrogen requirements into B31.8 (2026 edition)
- 3,072 miles of pre-1970 pipe in Minnesota alone, 209 miles of unknown vintage

### Governing Physics (5-mechanism sequential chain)

1. **Hydrogen diffusion PDE** (Oriani stress-assisted Fick's law):
   dC_L/dt = D_L * d2C_L/dx2 + D_L * (V_H/RT) * d/dx(C_L * dsigma_h/dx)
   BC: C(0) = S*sqrt(p_H2), C(w) = 0

2. **Pit growth at ERW seam** (power law from NG service):
   a_pit = k * f_seam * t^n

3. **Pit-to-crack transition** (Murakami + El Haddad):
   K_pit = 0.65 * sigma * sqrt(pi * sqrt(area)) >= K_th(C_H)

4. **Hydrogen-assisted fatigue crack growth** (modified Paris law):
   da/dN = C_paris * dK^m * [1 + alpha_H * (C_H/C_ref)^beta_H]

5. **Failure criterion**:
   K_max >= K_IC(C_H) = K_IC_air * exp(-lambda * C_H / C_ref)

### Repository Structure

```
integrity_code_series_week7_h2_lferw/
    src/
        config.py                    # All parameters, [ASSUMED] flags
        hydrogen_diffusion.py        # 1D radial H diffusion PDE
        pit_to_crack.py              # Murakami SIF, transition criteria
        ha_fcg.py                    # HA-FCG + integrated life engine
        monte_carlo.py               # MC + LHS parametric sweep
        ml_surrogate.py              # GBR surrogate, feature ranking
        cybersecurity.py             # STRIDE, audit chain, sensor integrity
        visualization.py             # All plots + animated GIF
    tests/
        test_suite.py                # 182 tests
    assets/                          # Generated visuals
    notebooks/                       # Optional Jupyter notebooks
    README.md
    requirements.txt
```

### Execution Order

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

### Key Assumptions Requiring Experimental Calibration

All parameters flagged [ASSUMED] in config.py. Critical ones:

- Sievert's constant S: governs surface hydrogen uptake
- Hydrogen embrittlement exponents (lambda_K, lambda_th): control property degradation rate
- HA-FCG enhancement parameters (alpha_H, beta_H): control crack growth acceleration
- ERW seam toughness K_IC and threshold K_th: must be measured on actual seam material in pressurized H2
- Seam enhancement factor f_seam: depends on specific manufacturing vintage

### Escalation from Previous Weeks

| Dimension | Week 5 (MIC) | Week 6 (Galvanic) | Week 7 (H2+LF-ERW) |
|---|---|---|---|
| Physics coupling | Reaction-diffusion ODE | 2D Laplace + BV | Transport PDE + fracture ODE |
| Mechanism chain | Single (biofilm + acid) | Single (galvanic) | 5-mechanism sequential |
| Probabilistic | Parametric sweep only | MCMC inverse | Monte Carlo 10,000 + LHS |
| Spatial | 1D chainage | 2D cross-section | 1D radial + through-wall |
| Regulatory | None | None | ASME B31.12 / PHMSA mapping |

### Cybersecurity

7 threats documented (STRIDE + data poisoning). SHA-256 hash-chain audit logging.
Sensor spoofing detection for ILI data. Physics monotonicity check for training data integrity.

### License

Educational and research use. Not for production engineering decisions without
experimental calibration of all [ASSUMED] parameters.
