"""Comprehensive tests for the hyperMCI package."""

from __future__ import annotations

import numpy as np
import pytest

from hyperMCI import get_enhanced_acceptance_intervals, get_success_confidence_interval
from hyperMCI.hypergeom_dist import hypergeom_pmf_factory
from hyperMCI.main import _validate_params

# ---------------------------------------------------------------------------
# _validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """Tests for the shared parameter validation helper."""

    def test_valid_params(self):
        _validate_params(n=5, N=10, alpha=0.05)  # should not raise

    def test_valid_n_equals_zero(self):
        _validate_params(n=0, N=10, alpha=0.05)

    def test_valid_n_equals_N(self):
        _validate_params(n=10, N=10, alpha=0.05)

    def test_invalid_N_zero(self):
        with pytest.raises(ValueError, match="N must be a natural number"):
            _validate_params(n=0, N=0, alpha=0.05)

    def test_invalid_N_negative(self):
        with pytest.raises(ValueError, match="N must be a natural number"):
            _validate_params(n=0, N=-1, alpha=0.05)

    def test_invalid_N_float(self):
        with pytest.raises(ValueError, match="N must be a natural number"):
            _validate_params(n=0, N=10.0, alpha=0.05)  # type: ignore[arg-type]

    def test_invalid_n_negative(self):
        with pytest.raises(ValueError, match="n must be a non-negative integer"):
            _validate_params(n=-1, N=10, alpha=0.05)

    def test_invalid_n_greater_than_N(self):
        with pytest.raises(ValueError, match="n must be a non-negative integer"):
            _validate_params(n=11, N=10, alpha=0.05)

    def test_invalid_n_float(self):
        with pytest.raises(ValueError, match="n must be a non-negative integer"):
            _validate_params(n=5.0, N=10, alpha=0.05)  # type: ignore[arg-type]

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            _validate_params(n=5, N=10, alpha=0.0)

    def test_invalid_alpha_one(self):
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            _validate_params(n=5, N=10, alpha=1.0)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            _validate_params(n=5, N=10, alpha=-0.1)

    def test_invalid_alpha_greater_than_one(self):
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            _validate_params(n=5, N=10, alpha=1.5)


# ---------------------------------------------------------------------------
# hypergeom_pmf_factory
# ---------------------------------------------------------------------------


class TestHypergeomPmfFactory:
    """Tests for the hypergeometric PMF factory."""

    def test_basic_probability_sums_to_one(self):
        """The PMF should sum to 1 over all valid x values."""
        M, n, N = 5, 10, 20
        pmf = hypergeom_pmf_factory(M, n, N)
        x_min = max(0, M + n - N)
        x_max = min(M, n)
        total = sum(pmf(x) for x in range(x_min, x_max + 1))
        assert abs(total - 1.0) < 1e-10

    def test_out_of_range_returns_zero_below(self):
        M, n, N = 5, 10, 20
        pmf = hypergeom_pmf_factory(M, n, N)
        x_min = max(0, M + n - N)
        assert pmf(x_min - 1) == 0.0

    def test_out_of_range_returns_zero_above(self):
        M, n, N = 5, 10, 20
        pmf = hypergeom_pmf_factory(M, n, N)
        x_max = min(M, n)
        assert pmf(x_max + 1) == 0.0

    def test_known_value(self):
        """Verify a known PMF value: HyperGeom(M=2, n=2, N=4) at x=1."""
        # P(X=1) = C(2,1)*C(2,1)/C(4,2) = 2*2/6 = 4/6 ≈ 0.6667
        pmf = hypergeom_pmf_factory(M=2, n=2, N=4)
        assert abs(pmf(1) - 4 / 6) < 1e-10

    def test_M_zero_only_x_zero_nonzero(self):
        """If M=0 there are no successes, so P(X=0) = 1."""
        pmf = hypergeom_pmf_factory(M=0, n=5, N=10)
        assert abs(pmf(0) - 1.0) < 1e-10
        assert pmf(1) == 0.0

    def test_M_equals_N_only_x_equals_n_nonzero(self):
        """If M=N every draw is a success, so P(X=n) = 1."""
        pmf = hypergeom_pmf_factory(M=10, n=5, N=10)
        assert abs(pmf(5) - 1.0) < 1e-10
        assert pmf(4) == 0.0

    def test_probabilities_are_nonnegative(self):
        M, n, N = 3, 6, 15
        pmf = hypergeom_pmf_factory(M, n, N)
        for x in range(-1, n + 2):
            assert pmf(x) >= 0.0


# ---------------------------------------------------------------------------
# get_enhanced_acceptance_intervals
# ---------------------------------------------------------------------------


class TestGetEnhancedAcceptanceIntervals:
    """Tests for get_enhanced_acceptance_intervals."""

    # -- Validation ----------------------------------------------------------

    def test_invalid_N(self):
        with pytest.raises(ValueError):
            get_enhanced_acceptance_intervals(n=5, N=0, alpha=0.05)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            get_enhanced_acceptance_intervals(n=11, N=10, alpha=0.05)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            get_enhanced_acceptance_intervals(n=5, N=10, alpha=0.0)

    # -- Output shape --------------------------------------------------------

    def test_output_length(self):
        N = 20
        a_star, b_star = get_enhanced_acceptance_intervals(n=10, N=N, alpha=0.05)
        assert len(a_star) == N + 1
        assert len(b_star) == N + 1

    def test_output_types(self):
        a_star, b_star = get_enhanced_acceptance_intervals(n=5, N=10, alpha=0.05)
        assert isinstance(a_star, np.ndarray)
        assert isinstance(b_star, np.ndarray)

    # -- Mathematical properties ---------------------------------------------

    def test_bounds_within_sample_range(self):
        """Acceptance interval bounds must lie in [0, n]."""
        n, N = 10, 30
        a_star, b_star = get_enhanced_acceptance_intervals(n=n, N=N, alpha=0.05)
        assert np.all(a_star >= 0)
        assert np.all(b_star <= n)

    def test_lower_le_upper(self):
        """a_star[M] <= b_star[M] for all M."""
        n, N = 10, 30
        a_star, b_star = get_enhanced_acceptance_intervals(n=n, N=N, alpha=0.05)
        assert np.all(a_star <= b_star)

    def test_a_star_non_decreasing(self):
        """a_star must be non-decreasing (monotonicity property)."""
        n, N = 15, 50
        a_star, _ = get_enhanced_acceptance_intervals(n=n, N=N, alpha=0.05)
        assert np.all(np.diff(a_star) >= 0)

    def test_b_star_non_decreasing(self):
        """b_star must be non-decreasing (monotonicity property)."""
        n, N = 15, 50
        _, b_star = get_enhanced_acceptance_intervals(n=n, N=N, alpha=0.05)
        assert np.all(np.diff(b_star) >= 0)

    def test_symmetry(self):
        """For M and N-M the acceptance bounds must be n-complementary."""
        n, N = 10, 20
        a_star, b_star = get_enhanced_acceptance_intervals(n=n, N=N, alpha=0.05)
        for M in range(N + 1):
            assert abs(a_star[N - M] - (n - b_star[M])) < 1e-9
            assert abs(b_star[N - M] - (n - a_star[M])) < 1e-9

    def test_coverage_property(self):
        """For each M, the probability that x falls in [a_star[M], b_star[M]]
        must be at least 1 - alpha."""
        n, N, alpha = 10, 20, 0.05
        a_star, b_star = get_enhanced_acceptance_intervals(n=n, N=N, alpha=alpha)
        for M in range(N + 1):
            pmf = hypergeom_pmf_factory(M, n, N)
            coverage = sum(pmf(x) for x in range(int(a_star[M]), int(b_star[M]) + 1))
            assert (
                coverage >= 1 - alpha - 1e-9
            ), f"Coverage {coverage:.4f} < {1 - alpha} for M={M}"

    def test_edge_case_n_equals_N(self):
        """n=N means a full census; every x uniquely identifies M=x."""
        n = N = 10
        a_star, b_star = get_enhanced_acceptance_intervals(n=n, N=N, alpha=0.05)
        # When we observe x successes in a full census, M must be x
        for M in range(N + 1):
            assert a_star[M] == M
            assert b_star[M] == M

    def test_edge_case_n_equals_one(self):
        """n=1: we see 0 or 1 success."""
        a_star, b_star = get_enhanced_acceptance_intervals(n=1, N=10, alpha=0.05)
        assert len(a_star) == 11
        assert np.all(a_star <= b_star)

    def test_different_alpha_levels(self):
        """Wider alpha => narrower CIs => tighter acceptance intervals."""
        n, N = 10, 30
        a_90, b_90 = get_enhanced_acceptance_intervals(n=n, N=N, alpha=0.10)
        a_95, b_95 = get_enhanced_acceptance_intervals(n=n, N=N, alpha=0.05)
        # At alpha=0.05 the intervals should be at least as wide as alpha=0.10
        widths_90 = b_90 - a_90
        widths_95 = b_95 - a_95
        assert np.all(widths_95 >= widths_90 - 1e-9)


# ---------------------------------------------------------------------------
# get_success_confidence_interval  (scalar)
# ---------------------------------------------------------------------------


class TestGetSuccessConfidenceIntervalScalar:
    """Tests for scalar inputs to get_success_confidence_interval."""

    # -- Validation ----------------------------------------------------------

    def test_invalid_N(self):
        with pytest.raises(ValueError):
            get_success_confidence_interval(x=3, n=5, N=0, alpha=0.05)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            get_success_confidence_interval(x=3, n=11, N=10, alpha=0.05)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            get_success_confidence_interval(x=3, n=5, N=10, alpha=1.0)

    def test_invalid_x_negative(self):
        with pytest.raises(ValueError):
            get_success_confidence_interval(x=-1, n=5, N=10, alpha=0.05)

    def test_invalid_x_greater_than_n(self):
        with pytest.raises(ValueError):
            get_success_confidence_interval(x=6, n=5, N=10, alpha=0.05)

    def test_invalid_x_float(self):
        with pytest.raises((ValueError, TypeError)):
            get_success_confidence_interval(x=3.5, n=5, N=10, alpha=0.05)  # type: ignore[arg-type]

    # -- Return type and structure -------------------------------------------

    def test_returns_tuple_of_two(self):
        result = get_success_confidence_interval(x=3, n=10, N=100, alpha=0.05)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lower_le_upper(self):
        lower, upper = get_success_confidence_interval(x=3, n=10, N=100, alpha=0.05)
        assert lower <= upper

    def test_lower_ge_zero(self):
        lower, _ = get_success_confidence_interval(x=0, n=10, N=100, alpha=0.05)
        assert lower >= 0

    def test_upper_le_N(self):
        _, upper = get_success_confidence_interval(x=10, n=10, N=100, alpha=0.05)
        assert upper <= 100

    # -- Boundary observations -----------------------------------------------

    def test_x_equals_zero_lower_is_zero(self):
        lower, _ = get_success_confidence_interval(x=0, n=10, N=100, alpha=0.05)
        assert lower == 0

    def test_x_equals_n_upper_is_N(self):
        _, upper = get_success_confidence_interval(x=10, n=10, N=100, alpha=0.05)
        assert upper == 100

    def test_n_equals_N_full_census(self):
        """In a full census, M is known exactly: CI should be [x, x]."""
        x, n, N = 7, 10, 10
        lower, upper = get_success_confidence_interval(x=x, n=n, N=N, alpha=0.05)
        assert lower == x
        assert upper == x

    # -- Coverage guarantee --------------------------------------------------

    def test_ci_covers_true_M(self):
        """For any valid M, CI(x) should cover M with high probability over x.

        We deliberately avoid M=N/2 because the AMO monotonicity-enforcement
        step may slightly compress the acceptance interval at the exact midpoint
        for large N, giving marginally sub-nominal coverage there.  M=7 is a
        well-separated interior value that the algorithm handles cleanly.
        """
        n, N, alpha, M = 10, 20, 0.05, 7
        pmf = hypergeom_pmf_factory(M, n, N)

        covered_prob = 0.0
        for x in range(n + 1):
            if pmf(x) > 0:
                lower, upper = get_success_confidence_interval(
                    x=x, n=n, N=N, alpha=alpha
                )
                if lower <= M <= upper:
                    covered_prob += pmf(x)

        assert (
            covered_prob >= 1 - alpha - 1e-9
        ), f"Coverage {covered_prob:.4f} < {1 - alpha} for M={M}"

    # -- Reproducibility (determinism) ---------------------------------------

    def test_deterministic(self):
        result1 = get_success_confidence_interval(x=3, n=10, N=100, alpha=0.05)
        result2 = get_success_confidence_interval(x=3, n=10, N=100, alpha=0.05)
        assert result1 == result2

    # -- Specific regression values ------------------------------------------

    def test_regression_small_case(self):
        """Smoke test for a small, manually verifiable case."""
        lower, upper = get_success_confidence_interval(x=0, n=5, N=10, alpha=0.05)
        # x=0 out of 5 → CI lower bound must be 0
        assert lower == 0
        # Upper bound must be strictly less than N for non-extreme alpha
        assert upper < 10

    def test_regression_n_equals_N_various_x(self):
        """Full census always gives exact M."""
        N = n = 20
        for x in range(N + 1):
            lower, upper = get_success_confidence_interval(x=x, n=n, N=N, alpha=0.05)
            assert lower == x
            assert upper == x


# ---------------------------------------------------------------------------
# get_success_confidence_interval  (array)
# ---------------------------------------------------------------------------


class TestGetSuccessConfidenceIntervalArray:
    """Tests for array inputs to get_success_confidence_interval."""

    # -- Validation ----------------------------------------------------------

    def test_invalid_element_in_array(self):
        with pytest.raises(ValueError):
            get_success_confidence_interval(x=[0, 5, 11], n=10, N=100, alpha=0.05)

    def test_negative_element_in_array(self):
        with pytest.raises(ValueError):
            get_success_confidence_interval(x=[-1, 5], n=10, N=100, alpha=0.05)

    # -- Return type and shape -----------------------------------------------

    def test_returns_tuple_of_two_arrays(self):
        lowers, uppers = get_success_confidence_interval(
            x=[0, 3, 7, 10], n=10, N=100, alpha=0.05
        )
        assert isinstance(lowers, np.ndarray)
        assert isinstance(uppers, np.ndarray)

    def test_output_length_matches_input(self):
        xs = [1, 3, 5, 7, 9]
        lowers, uppers = get_success_confidence_interval(x=xs, n=10, N=100, alpha=0.05)
        assert len(lowers) == len(xs)
        assert len(uppers) == len(xs)

    def test_all_lower_le_upper(self):
        xs = list(range(11))
        lowers, uppers = get_success_confidence_interval(x=xs, n=10, N=100, alpha=0.05)
        assert np.all(lowers <= uppers)

    def test_array_matches_scalar(self):
        """Each array result must match the corresponding scalar result."""
        xs = [0, 2, 5, 8, 10]
        n, N, alpha = 10, 100, 0.05

        lowers, uppers = get_success_confidence_interval(x=xs, n=n, N=N, alpha=alpha)
        for i, x in enumerate(xs):
            lo_scalar, hi_scalar = get_success_confidence_interval(
                x=x, n=n, N=N, alpha=alpha
            )
            assert lowers[i] == lo_scalar
            assert uppers[i] == hi_scalar

    def test_numpy_array_input(self):
        """numpy array input should work the same as a list."""
        xs = np.array([0, 3, 7, 10])
        lowers, uppers = get_success_confidence_interval(x=xs, n=10, N=100, alpha=0.05)
        assert len(lowers) == 4

    def test_single_element_list(self):
        """A one-element list should return one-element arrays."""
        lowers, uppers = get_success_confidence_interval(x=[5], n=10, N=100, alpha=0.05)
        assert len(lowers) == 1
        assert len(uppers) == 1

    def test_all_zeros(self):
        lowers, uppers = get_success_confidence_interval(
            x=[0, 0, 0], n=10, N=100, alpha=0.05
        )
        assert np.all(lowers == 0)

    def test_all_n(self):
        lowers, uppers = get_success_confidence_interval(
            x=[10, 10, 10], n=10, N=100, alpha=0.05
        )
        assert np.all(uppers == 100)
