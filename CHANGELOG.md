# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
