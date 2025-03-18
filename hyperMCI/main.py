import math

import numpy as np

from .hypergeom_dist import hypergeom_pmf_factory


def _get_alpha_max_optimal_acceptance_intervals(n, N, alpha=0.05):
    """Calculate the confidence interval for a hypergeometric distribution.

    This function computes the confidence interval of number of "successes" M
    using the hypergeometric distribution given the population size,
    sample size, and significance level.

    Args:
        n (int): The sample size. Must be a non-negative integer less than or
            equal to N.
        N (int): The size of the population. Must be a positive integer.
        alpha (float): Corresponds to 1 - alpha significance level.
            Must be between 0 and 1 (exclusive).
            A 95% confidence level is the default value.

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence
        interval.

    Raises:
        ValueError: If n is not a non-negative integer less than or equal to N.
        ValueError: If N is not a positive integer.
        ValueError: If alpha is not in the interval (0, 1).
    """
    if not (isinstance(n, int) and 0 <= n <= N):
        raise ValueError(
            "n must be a non-negative integer less than or equal to N.")
    if not (isinstance(N, int) and N > 0):
        raise ValueError("N must be a natural number (a positive integer).")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive).")

    a = np.zeros(N + 1)
    b = np.zeros(N + 1)

    # Loop over M values
    for M in range(0, math.floor(N / 2) + 1):

        # Calculate initial boundaries
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


def get_enhanced_acceptance_intervals(n, N, alpha):
    """Calculate the confidence interval for a hypergeometric distribution.

    This function computes the confidence interval of number of "successes" M
    using the hypergeometric distribution given the population size,
    sample size, and significance level.

    Args:
        n (int): The sample size. Must be a non-negative integer less than or
        N (int): The size of the population. Must be a positive integer.
        equal to N.
        alpha (float): Corresponds to 1 - alpha significance level.
            Must be between 0 and 1 (exclusive).

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence
        interval.

    Raises:
        ValueError: If n is not a non-negative integer less than or equal to N.
        ValueError: If N is not a positive integer.
        ValueError: If alpha is not in the interval (0, 1).
    """
    if not (isinstance(n, int) and 0 <= n <= N):
        raise ValueError(
            "n must be a non-negative integer less than or equal to N.")
    if not (isinstance(N, int) and N > 0):
        raise ValueError("N must be a natural number (a positive integer).")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive).")

    # Declare a_star and b_start as numpy arrays of length N+1
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


def get_success_confidence_interval(x, n, N, alpha):
    """Calculate the confidence interval for the success variable M in a
    hypergeometric distribution.

    Args:
        x (int or array-like): Observed number of successes. Can be an
        integer between 0 and n inclusive or an array of such integers.
        n (int): The sample size. Must be a non-negative integer less than
        or equal to N.
        N (int): The size of the population. Must be a positive integer.
        alpha (float): Corresponds to 1 - alpha significance level.
            Must be between 0 and 1 (exclusive).

    Returns:
            If x is a single integer: A tuple (Lx, Ux) containing the lower
            and upper confidence interval bounds.
            If x is an array: A tuple (Lx_array, Ux_array) containing arrays
            of lower and upper confidence interval bounds for each element in
            x.
    """

    # Check if N is a positive integer
    if not (isinstance(N, int) and N > 0):
        raise ValueError("N must be a positive integer.")
    # Check if n is a non-negative integer less than or equal to N
    if not (isinstance(n, int) and 0 <= n <= N):
        raise ValueError(
            "n must be a non-negative integer less than or equal to N.")
    # Check if alpha is in the interval (0, 1)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive).")

    # Get amo intervals
    a_star, b_star = get_enhanced_acceptance_intervals(n, N, alpha)

    # Check if x is a single integer or an array
    is_array = hasattr(x, '__iter__') and not isinstance(x, (str, bytes))

    if is_array:
        # Convert x to numpy array if it's not already
        x_array = np.array(x, dtype=int)

        # Check if all elements of x are integers between 0 and n inclusive
        if not np.all((x_array >= 0) & (x_array <= n)):
            raise ValueError(
                "All elements of x must be integers between 0 "
                "and n inclusive."
            )

        # Initialize arrays for lower and upper confidence interval bounds
        Lx_array = np.zeros_like(x_array)
        Ux_array = np.zeros_like(x_array)

        # Calculate confidence intervals for each element in x
        for i, xi in enumerate(x_array):
            S = [M for M in range(N + 1) if a_star[M] <= xi <= b_star[M]]
            Lx_array[i] = min(S)
            Ux_array[i] = max(S)

        return Lx_array, Ux_array
    else:
        # Check if x is an integer between 0 and n inclusive
        if not (isinstance(x, int) and 0 <= x <= n):
            raise ValueError(
                "x must be an integer between 0 "
                "and n inclusive.")

        # Invert acceptance intervals for single integer x
        S = [M for M in range(N + 1) if a_star[M] <= x <= b_star[M]]
        Lx = min(S)
        Ux = max(S)

        return Lx, Ux
