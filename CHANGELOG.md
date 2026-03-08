# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-01

### Added
- Initial implementation of the Alpha-Max-Optimal (AMO) acceptance interval
  algorithm for the hypergeometric distribution, based on Bartroff et al. (2022).
- `get_enhanced_acceptance_intervals(n, N, alpha)` — computes the monotone
  AMO acceptance region for all M in {0, …, N}.
- `get_success_confidence_interval(x, n, N, alpha)` — inverts the acceptance
  region to return a (1 − alpha) confidence interval for M. Supports both
  scalar and array-like `x`.
- `hypergeom_pmf_factory` — efficient factory for pre-computing the
  hypergeometric PMF with a fixed (M, n, N) parameter set.
- MIT license.

[Unreleased]: https://github.com/oresthes/hyperMCI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/oresthes/hyperMCI/releases/tag/v0.1.0
