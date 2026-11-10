# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- LHS censoring: the sweep stopped at 100,000 cycles (274 yr), below the 279 yr default case, so 723 of 2,000 runs (154 of the 390 held-out runs) sat at the cap and Spearman rho and the surrogate were fitted to the capped value. The cap is now 1,000,000 cycles (2,740 yr): 43 runs (2.2%) censored. Censored runs are flagged and left out of the surrogate. Spearman rho: p_H2 -0.832 -> -0.855, pit depth -0.431 -> -0.460, K_IC +0.117 -> +0.109, c/a +0.026 -> +0.025, D_L +0.001 -> +0.004. Surrogate R^2 0.997 -> 0.993, MAE 3.5 -> 19.0 yr (lives now span 0 to 2,740 yr). Tornado, parity and iso-risk figures regenerated.
- Sign of the stress-drift term in the diffusion PDE: hydrogen now moves toward tension. With a linear test field C(w/2)/C0 goes 0.3775 -> 0.6225. No effect at the default Lame field, where sigma_h is uniform.
- `f_seam` was sampled and never used. It now scales pit depth in the Monte Carlo and is removed from the LHS sweep, which samples pit depth directly (6 -> 5 parameters).
- Pit-to-crack transition compares the threshold with the stress range: at a = 0.15 mm, K 5.25 -> dK 2.63 MPa sqrt(m); El Haddad a_th 0.040 -> 0.159 mm.
- `config.maop_h2` removed: it gave 4.475 MPa beside 2.417 MPa from `maop_under_b3112`.
- `plot_fad` used a hardcoded 7.0 MPa.
- FAD curve relabelled API 579-1 Level 2: the formula is that curve, not the BS 7910 Option 1 curve it was labelled as.
- All figures regenerated at the documented sample counts. The published Monte Carlo figure was a 200-sample run: P50 112 -> 189 yr, fraction below 10 yr 15% -> 0.8%.

### Changed
- `F` and `H_f` are tagged `[ASSUMED]`: the cited B31.12 table could not be verified
- Oriani attribution removed: the model has no trapping
- News items that could not be verified removed from the README

### Docs
- docs: insert real Zenodo DOIs (concept 10.5281/zenodo.20172481, version 10.5281/zenodo.20172482)

### CI
- ci: drop Python 3.10 from matrix (modern scipy/numpy require >=3.11)
- ci: let test failures fail the job (remove `|| true`); bump checkout and setup-python to v7
- ci: Dependabot keeps pip floors unless required and groups github-actions updates

### Added
- `run_all.py`: full chain and all figures from one command, with a `--quick` smoke run

### Removed
- `assets/surrogate_model.pkl`: unused, and no longer loadable with the required scikit-learn
- NACE TM0316 from the metadata: no part of this package implements it

## [1.0.1] - 2026-05-13

### Changed
- chore(release): prep v1.0.1 — Zenodo metadata, DOI badge, How to Cite
- chore(meta): add CI matrix, Dependabot, and Integrity Code Series cross-links

[1.0.1]: https://github.com/felipearocha/integrity-code-series-week7-h2-lferw/releases/tag/v1.0.1

## [1.0.0] - 2026-03-31

### Other
- Week 7: Hydrogen conversion of aging LF-ERW pipeline
- Initial commit

[1.0.0]: https://github.com/felipearocha/integrity-code-series-week7-h2-lferw/releases/tag/v1.0.0
