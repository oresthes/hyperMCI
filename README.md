# hyperMCI

[![CI](https://github.com/oresthes/hyperMCI/actions/workflows/ci.yml/badge.svg)](https://github.com/oresthes/hyperMCI/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/hyperMCI.svg)](https://badge.fury.io/py/hyperMCI)
[![Python Versions](https://img.shields.io/pypi/pyversions/hyperMCI.svg)](https://pypi.org/project/hyperMCI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for calculating optimal confidence intervals for the hypergeometric distribution success parameter.

## Description

**hyperMCI** implements the Alpha-Max-Optimal (AMO) confidence interval method described by Bartroff et al. (2022) in *"Optimal and fast confidence intervals for hypergeometric successes"*. It provides efficient algorithms for computing confidence intervals with guaranteed coverage probability for the hypergeometric distribution — useful in audit sampling, clinical trials, quality control, and survey statistics.

## Installation

```bash
pip install hyperMCI
```

Or, if you use [Poetry](https://python-poetry.org/):

```bash
poetry add hyperMCI
```

## Quick Start

```python
from hyperMCI import get_success_confidence_interval

# 95% CI for M given x=3 successes in a sample of n=10 from N=100
lower, upper = get_success_confidence_interval(x=3, n=10, N=100, alpha=0.05)
print(f"95% Confidence interval for M: [{lower}, {upper}]")
```

## Usage

### Single observation

```python
from hyperMCI import get_success_confidence_interval

lower, upper = get_success_confidence_interval(x=5, n=20, N=200, alpha=0.05)
print(f"95% CI: [{lower}, {upper}]")
```

### Multiple observations at once

```python
import numpy as np
from hyperMCI import get_success_confidence_interval

observations = [2, 5, 8, 10]
lowers, uppers = get_success_confidence_interval(x=observations, n=20, N=200, alpha=0.05)

for x, lo, hi in zip(observations, lowers, uppers):
    print(f"x={x}: CI=[{lo}, {hi}]")
```

### Access the acceptance intervals directly

```python
from hyperMCI import get_enhanced_acceptance_intervals

# a_star[M] and b_star[M] give the acceptance interval bounds for each M
a_star, b_star = get_enhanced_acceptance_intervals(n=20, N=200, alpha=0.05)
```

## Features

- **Optimal CIs**: Implements the AMO method for shortest possible confidence intervals while maintaining the desired coverage probability.
- **Scalar and batch support**: `get_success_confidence_interval` accepts a single integer or any array-like input.
- **Minimal dependencies**: Only requires NumPy.
- **Fully typed**: Ships with a `py.typed` marker for PEP 561 compliance.

## Mathematical Background

The hypergeometric distribution models sampling without replacement from a finite population. Given a population of size N containing M successes, the probability of observing x successes in a sample of size n follows:

```
P(X = x | M, n, N) = C(M, x) * C(N-M, n-x) / C(N, n)
```

This package implements the Alpha-Max-Optimal (AMO) confidence intervals described by Bartroff et al. (2022), which provide shorter intervals while maintaining the desired coverage probability compared to classical methods (e.g., Clopper-Pearson adapted to the hypergeometric setting).

## Citation

If you use this package in your research, please cite:

```bibtex
@article{bartroff2022optimal,
  title   = {Optimal and fast confidence intervals for hypergeometric successes},
  author  = {Bartroff, Jay and others},
  year    = {2022}
}
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
