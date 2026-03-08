from __future__ import annotations

import math
from typing import Union

import numpy as np

from .hypergeom_dist import hypergeom_pmf_factory


def _validate_params(n: int, N: int, alpha: float) -> None:
    """Validate parameters shared across all public functions.

    Args:
        n: The sample size.
        N: The population size.
        alpha: The significance level.

    Raises:
        ValueError: If N is not a positive integer.
        ValueError: If n is not a non-negative integer less than or equal to N.
        ValueError: If alpha is not in the interval (0, 1).
    """
    if not (isinstance(N, int) and N > 0):
        raise ValueError("N must be a natural number (a positive integer).")
    if not (isinstance(n, int) and 0 <= n <= N):
        raise ValueError(
            "n must be a non-negative integer less than or equal to N."
        )
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive).")


def _get_alpha_max_optimal_acceptance_intervals(
    n: int, N: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the AMO acceptance intervals for a hypergeometric distribution.

    This private helper computes the Alpha-Max-Optimal (AMO) acceptance
    intervals for the number of successes M. Parameters are assumed to have
    been validated by the caller.

    Args:
        n: The sample size (non-negative integer, n <= N).
        N: The size of the population (positive integer).
        alpha: The significance level in (0, 1). Defaults to 0.05.

    Returns:
        A tuple (a, b) of numpy arrays of length N+1 where a[M] and b[M]
        are the lower and upper acceptance interval bounds for each M.
    """
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)

    for M in range(0, math.floor(N / 2) + 1):
        C = D = math.floor(((n + 1) * (M + 1)) / (N + 2))

        PM = hypergeom_pmf_factory(M, n, N)
        P = PM(C)

        PC = PM(C - 1)
        PD = PM(D + 1)

        while P < 1 - alpha:
            if PD > PC:
                D += 1
                P += PD
                PD = PM(D + 1)
            else:
                C -= 1
                P += PC
                PC = PM(C - 1)

        a[M] = C
        b[M] = D
        a[N - M] = n - b[M]
        b[N - M] = n - a[M]

    return a, b


def get_enhanced_acceptance_intervals(
    n: int, N: int, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the enhanced AMO acceptance intervals for a hypergeometric
    distribution.

    This function computes the enhanced Alpha-Max-Optimal (AMO) acceptance
    intervals for the number of successes M using the hypergeometric
    distribution, given the population size, sample size, and significance
    level. The enhancement enforces monotonicity of acceptance region
    boundaries, which is required for valid confidence interval inversion.

    Args:
        n (int): The sample size. Must be a non-negative integer less than
            or equal to N.
        N (int): The size of the population. Must be a positive integer.
        alpha (float): The significance level. Corresponds to a
            (1 - alpha) confidence level. Must be between 0 and 1
            (exclusive). For a 95% confidence interval use alpha=0.05.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple (a_star, b_star) of numpy
            arrays of length N+1, where a_star[M] and b_star[M] are the
            lower and upper bounds of the enhanced AMO acceptance interval
            for each M.

    Raises:
        ValueError: If N is not a positive integer.
        ValueError: If n is not a non-negative integer less than or equal to N.
        ValueError: If alpha is not in the interval (0, 1).

    Example:
        >>> a_star, b_star = get_enhanced_acceptance_intervals(n=10, N=100, alpha=0.05)
        >>> # a_star[M] and b_star[M] give the acceptance bounds for each M
    """
    _validate_params(n, N, alpha)

    a_star = np.zeros(N + 1)
    b_star = np.zeros(N + 1)

    a, b = _get_alpha_max_optimal_acceptance_intervals(n, N, alpha)

    a_star[0] = a[0]
    b_star[0] = b[0]
    for M in range(1, math.floor(N / 2) + 1):
        if a[M] < a_star[M - 1]:
            a_star[M] = a_star[M - 1]
            b_star[M] = b[M] + a_star[M - 1] - a[M]
        else:
            b_star[M] = b[M]
            a_star[M] = a[M]

    for M in range(math.floor(N / 2) - 1, -1, -1):
        if b_star[M] > b_star[M + 1]:
            a_star[M] = a_star[M] + b_star[M + 1] - b_star[M]
            b_star[M] = b_star[M + 1]
        a_star[N - M] = n - b_star[M]
        b_star[N - M] = n - a_star[M]

    if N % 2 == 0:
        a_star[N // 2] = max(a[N // 2], n - b[N // 2])
        b_star[N // 2] = n - a_star[N // 2]
    else:
        a_star[math.floor(N / 2) + 1] = n - b_star[math.floor(N / 2)]
        b_star[math.floor(N / 2) + 1] = n - a_star[math.floor(N / 2)]

    return a_star, b_star


def get_success_confidence_interval(
    x: Union[int, list[int], np.ndarray],
    n: int,
    N: int,
    alpha: float,
) -> Union[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Calculate the confidence interval for the success count M in a
    hypergeometric distribution.

    Given x observed successes in a sample of size n drawn without
    replacement from a population of N, this function returns the
    (1 - alpha) confidence interval for the unknown total number of
    successes M in the population.

    Args:
        x (int or array-like): Observed number of successes in the sample.
            Must be an integer between 0 and n inclusive, or an array of
            such integers.
        n (int): The sample size. Must be a non-negative integer less than
            or equal to N.
        N (int): The size of the population. Must be a positive integer.
        alpha (float): The significance level. Must be between 0 and 1
            (exclusive). For a 95% confidence interval use alpha=0.05.

    Returns:
        If x is a single integer: a tuple (Lx, Ux) of ints representing
            the lower and upper confidence bounds for M.
        If x is array-like: a tuple (Lx_array, Ux_array) of numpy arrays
            containing the lower and upper confidence bounds for each
            element of x.

    Raises:
        ValueError: If N is not a positive integer.
        ValueError: If n is not a non-negative integer less than or equal
            to N.
        ValueError: If alpha is not in the interval (0, 1).
        ValueError: If x (or any element of x) is not between 0 and n.

    Example:
        >>> lower, upper = get_success_confidence_interval(x=3, n=10, N=100, alpha=0.05)
        >>> print(f"95% CI for M: [{lower}, {upper}]")
        95% CI for M: [6, 57]
    """
    _validate_params(n, N, alpha)

    a_star, b_star = get_enhanced_acceptance_intervals(n, N, alpha)

    is_array = hasattr(x, "__iter__") and not isinstance(x, (str, bytes))

    if is_array:
        x_array = np.array(x, dtype=int)

        if not np.all((x_array >= 0) & (x_array <= n)):
            raise ValueError(
                "All elements of x must be integers between 0 and n inclusive."
            )

        Lx_array = np.zeros_like(x_array)
        Ux_array = np.zeros_like(x_array)

        for i, xi in enumerate(x_array):
            S = [M for M in range(N + 1) if a_star[M] <= xi <= b_star[M]]
            if not S:
                raise ValueError(
                    f"No valid M found for observed x={xi}. "
                    "This indicates a bug — please open an issue at "
                    "https://github.com/oresthes/hyperMCI/issues."
                )
            Lx_array[i] = min(S)
            Ux_array[i] = max(S)

        return Lx_array, Ux_array

    else:
        if not (isinstance(x, int) and 0 <= x <= n):
            raise ValueError(
                "x must be an integer between 0 and n inclusive."
            )

        S = [M for M in range(N + 1) if a_star[M] <= x <= b_star[M]]
        if not S:
            raise ValueError(
                f"No valid M found for observed x={x}. "
                "This indicates a bug — please open an issue at "
                "https://github.com/oresthes/hyperMCI/issues."
            )
        Lx = min(S)
        Ux = max(S)

        return Lx, Ux
