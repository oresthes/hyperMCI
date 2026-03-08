from __future__ import annotations

import math
from functools import partial
from typing import Callable


def _hypergeom_pmf_with_range(
    x: int, M: int, n: int, N: int, x_min: int, x_max: int
) -> float:
    """Base PMF function that expects x_min/x_max to be precomputed."""
    # Return 0 if x is outside the valid range
    if x < x_min or x > x_max:
        return 0.0

    return (math.comb(M, x) * math.comb(N - M, n - x)) / math.comb(N, n)


def hypergeom_pmf_factory(M: int, n: int, N: int) -> Callable[[int], float]:
    """Create a partial function for Hypergeometric PMF with fixed (M, N, n).

    The valid range for x is:
        x_min = max(0, M + n - N)
        x_max = min(M, n)
    Values of x outside this range automatically return 0.

    Args:
        M (int): Number of "successes" in the population (0 <= M <= N).
        n (int): Sample size (0 <= n <= N).
        N (int): Total population size.

    Returns:
        Callable[[int], float]: A function pmf(x) -> float.

    Example:
        # Suppose M=10 successes, N=100 total, n=5 draws
        pmf = hypergeom_pmf_factory(M=10, N=100, n=5)
        pmf(3)  # Probability that exactly 3 out of 5 draws are successes
    """
    # Precompute valid x-range for the hypergeometric distribution
    x_min = max(0, M + n - N)  # e.g. if M+n <= N, x_min=0
    x_max = min(M, n)

    # Return a partial function that only needs `x`
    return partial(_hypergeom_pmf_with_range, M=M, n=n, N=N, x_min=x_min, x_max=x_max)
