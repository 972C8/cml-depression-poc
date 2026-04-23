"""Unit tests for fuzzy membership functions.

Tests triangular and trapezoidal membership functions for edge cases,
boundary conditions, and parameter validation.
"""

import pytest

from src.core.config import MembershipFunction
from src.core.context.membership import LinguisticVariable, MembershipCalculator


class TestMembershipCalculatorTriangular:
    """Tests for triangular membership function."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.calculator = MembershipCalculator()

    def test_triangular_at_left_boundary_returns_zero(self) -> None:
        """Test triangular at left boundary (x = left) -> 0."""
        result = self.calculator.triangular(x=0.0, left=0.0, peak=5.0, right=10.0)
        assert result == 0.0

    def test_triangular_at_peak_returns_one(self) -> None:
        """Test triangular at peak (x = peak) -> 1.0."""
        result = self.calculator.triangular(x=5.0, left=0.0, peak=5.0, right=10.0)
        assert result == 1.0

    def test_triangular_at_right_boundary_returns_zero(self) -> None:
        """Test triangular at right boundary (x = right) -> 0."""
        result = self.calculator.triangular(x=10.0, left=0.0, peak=5.0, right=10.0)
        assert result == 0.0

    def test_triangular_midpoint_left_slope(self) -> None:
        """Test triangular at midpoint of left slope."""
        # x=2.5 on left slope from 0 to 5 should give 0.5
        result = self.calculator.triangular(x=2.5, left=0.0, peak=5.0, right=10.0)
        assert result == pytest.approx(0.5)

    def test_triangular_midpoint_right_slope(self) -> None:
        """Test triangular at midpoint of right slope."""
        # x=7.5 on right slope from 5 to 10 should give 0.5
        result = self.calculator.triangular(x=7.5, left=0.0, peak=5.0, right=10.0)
        assert result == pytest.approx(0.5)

    def test_triangular_outside_range_left_returns_zero(self) -> None:
        """Test values left of range return 0."""
        result = self.calculator.triangular(x=-5.0, left=0.0, peak=5.0, right=10.0)
        assert result == 0.0

    def test_triangular_outside_range_right_returns_zero(self) -> None:
        """Test values right of range return 0."""
        result = self.calculator.triangular(x=15.0, left=0.0, peak=5.0, right=10.0)
        assert result == 0.0

    def test_triangular_parameter_validation_raises_on_invalid(self) -> None:
        """Test parameter validation (left < peak < right)."""
        with pytest.raises(ValueError, match="Invalid triangular parameters"):
            self.calculator.triangular(x=5.0, left=10.0, peak=5.0, right=0.0)

    def test_triangular_degenerate_left_peak_equal(self) -> None:
        """Test degenerate case where left == peak."""
        result = self.calculator.triangular(x=0.0, left=0.0, peak=0.0, right=5.0)
        assert result == 0.0

    def test_triangular_degenerate_peak_right_equal(self) -> None:
        """Test degenerate case where peak == right."""
        result = self.calculator.triangular(x=5.0, left=0.0, peak=5.0, right=5.0)
        assert result == 0.0


class TestMembershipCalculatorTrapezoidal:
    """Tests for trapezoidal membership function."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.calculator = MembershipCalculator()

    def test_trapezoidal_flat_top_region_returns_one(self) -> None:
        """Test trapezoidal flat top region -> 1.0."""
        # Between left_peak and right_peak should be 1.0
        result = self.calculator.trapezoidal(
            x=5.0, left=0.0, left_peak=3.0, right_peak=7.0, right=10.0
        )
        assert result == 1.0

    def test_trapezoidal_left_slope(self) -> None:
        """Test trapezoidal on left slope."""
        # x=1.5 on left slope from 0 to 3 should give 0.5
        result = self.calculator.trapezoidal(
            x=1.5, left=0.0, left_peak=3.0, right_peak=7.0, right=10.0
        )
        assert result == pytest.approx(0.5)

    def test_trapezoidal_right_slope(self) -> None:
        """Test trapezoidal on right slope."""
        # x=8.5 on right slope from 7 to 10 should give 0.5
        result = self.calculator.trapezoidal(
            x=8.5, left=0.0, left_peak=3.0, right_peak=7.0, right=10.0
        )
        assert result == pytest.approx(0.5)

    def test_trapezoidal_at_left_boundary_returns_zero(self) -> None:
        """Test trapezoidal at left boundary -> 0."""
        result = self.calculator.trapezoidal(
            x=0.0, left=0.0, left_peak=3.0, right_peak=7.0, right=10.0
        )
        assert result == 0.0

    def test_trapezoidal_at_right_boundary_returns_zero(self) -> None:
        """Test trapezoidal at right boundary -> 0."""
        result = self.calculator.trapezoidal(
            x=10.0, left=0.0, left_peak=3.0, right_peak=7.0, right=10.0
        )
        assert result == 0.0

    def test_trapezoidal_outside_range_returns_zero(self) -> None:
        """Test values outside range return 0."""
        result = self.calculator.trapezoidal(
            x=-5.0, left=0.0, left_peak=3.0, right_peak=7.0, right=10.0
        )
        assert result == 0.0

    def test_trapezoidal_parameter_validation_raises_on_invalid(self) -> None:
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Invalid trapezoidal parameters"):
            self.calculator.trapezoidal(
                x=5.0, left=10.0, left_peak=7.0, right_peak=3.0, right=0.0
            )


class TestMembershipCalculatorDispatcher:
    """Tests for the calculate() dispatcher method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.calculator = MembershipCalculator()

    def test_calculate_triangular(self) -> None:
        """Test calculate dispatches to triangular correctly."""
        fn = MembershipFunction(type="triangular", params=[0.0, 5.0, 10.0])
        result = self.calculator.calculate(x=5.0, membership_fn=fn)
        assert result == 1.0

    def test_calculate_trapezoidal(self) -> None:
        """Test calculate dispatches to trapezoidal correctly."""
        fn = MembershipFunction(type="trapezoidal", params=[0.0, 3.0, 7.0, 10.0])
        result = self.calculator.calculate(x=5.0, membership_fn=fn)
        assert result == 1.0

    def test_calculate_unknown_type_raises(self) -> None:
        """Test calculate raises on unknown function type."""
        # Create a mock with invalid type
        fn = MembershipFunction(type="triangular", params=[0.0, 5.0, 10.0])
        # Manually override type to test error handling
        object.__setattr__(fn, "type", "unknown")
        with pytest.raises(ValueError, match="Unknown membership function type"):
            self.calculator.calculate(x=5.0, membership_fn=fn)


class TestLinguisticVariable:
    """Tests for LinguisticVariable class."""

    def test_evaluate_returns_all_set_memberships(self) -> None:
        """Test evaluate returns membership for all defined sets."""
        var = LinguisticVariable(
            name="people_in_room",
            membership_fns={
                "low": MembershipFunction(type="triangular", params=[0, 0, 2]),
                "medium": MembershipFunction(type="triangular", params=[1, 3, 5]),
                "high": MembershipFunction(type="trapezoidal", params=[4, 6, 10, 10]),
            },
        )

        # Value of 3.5 should have overlapping membership in medium and high
        result = var.evaluate(3.5)

        assert "low" in result
        assert "medium" in result
        assert "high" in result
        assert result["low"] == 0.0  # Outside low range
        assert result["medium"] == pytest.approx(0.75)  # On right slope of medium
        assert result["high"] == 0.0  # Just below high range

    def test_evaluate_handles_overlapping_membership(self) -> None:
        """Test value can belong to multiple sets."""
        var = LinguisticVariable(
            name="test",
            membership_fns={
                "low": MembershipFunction(type="triangular", params=[0, 2, 4]),
                "high": MembershipFunction(type="triangular", params=[2, 4, 6]),
            },
        )

        # At x=3, should have partial membership in both
        result = var.evaluate(3.0)
        assert result["low"] == pytest.approx(0.5)
        assert result["high"] == pytest.approx(0.5)

    def test_evaluate_empty_membership_fns(self) -> None:
        """Test evaluate with no membership functions."""
        var = LinguisticVariable(name="empty", membership_fns={})
        result = var.evaluate(5.0)
        assert result == {}
