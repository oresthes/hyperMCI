# hyperMCI

A Python package for calculating optimal confidence intervals for the hypergeometric distribution success parameter.

## Description

hyperMCI implements the optimal confidence interval calculation methods described by Bartroff et al. (2022) in their paper "Optimal and fast confidence intervals for hypergeometric successes". This package provides efficient algorithms for computing confidence intervals with guaranteed coverage probability for the hypergeometric distribution.

## Installation

````bash
# Install using pip
# if available in Pypi or private repository
pip install hyperMCI
# or
pip install git+https://gitlab.com/mintel/dsa/tools/cpg/hypermci.git

# Or using Poetry
# if available in Pypi or private repository
poetry add hyperMCI 
# or
poetry add git+https://gitlab.com/mintel/dsa/tools/cpg/hypermci.git
````

## Usage

```python
from hyperMCI import CI_interval

# Calculate 95% confidence interval
# Parameters: (observed successes, sample size, population size, significance level)
lower, upper = get_success_confidence_interval(x=3, n=10, N=100, alpha=0.05)
print(f"95% Confidence interval: [{lower}, {upper}]")
```


## Features

- Calculate confidence intervals for hypergeometric distributions
- Implementation of the Alpha-Max-Optimal (AMO) approach for optimal acceptance intervals
- Efficient calculation of hypergeometric probability mass functions


## Mathematical Background

The hypergeometric distribution models sampling without replacement from a finite population. Given a population of size N containing M successes, the probability of observing x successes in a sample of size n follows the hypergeometric distribution.

This package implements the alpha-max optimal (AMO) confidence intervals described by Bartroff et al., which provide shorter intervals while maintaining the desired coverage probability.

## Citation

If you use this package in your research, please cite:

```
Bartroff, J., et al. (2022). Optimal and fast confidence intervals for hypergeometric successes.
```

## License

MIT

