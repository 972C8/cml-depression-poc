"""Tests for biomarker membership functions."""

import math

import pytest

from src.core.processors.membership import BiomarkerMembershipCalculator


class TestBiomarkerMembershipCalculator:
    """Test BiomarkerMembershipCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return BiomarkerMembershipCalculator()

    def test_triangular_peak(self, calculator):
        """Test triangular function at peak."""
        result = calculator.triangular(z=0.0, left=-2.0, m=0.0, h=2.0)
        assert result == 1.0

    def test_triangular_left_boundary(self, calculator):
        """Test triangular function at left boundary."""
        result = calculator.triangular(z=-2.0, left=-2.0, m=0.0, h=2.0)
        assert result == 0.0

    def test_triangular_right_boundary(self, calculator):
        """Test triangular function at right boundary."""
        result = calculator.triangular(z=2.0, left=-2.0, m=0.0, h=2.0)
        assert result == 0.0

    def test_triangular_outside_left(self, calculator):
        """Test triangular function outside left boundary."""
        result = calculator.triangular(z=-3.0, left=-2.0, m=0.0, h=2.0)
        assert result == 0.0

    def test_triangular_outside_right(self, calculator):
        """Test triangular function outside right boundary."""
        result = calculator.triangular(z=3.0, left=-2.0, m=0.0, h=2.0)
        assert result == 0.0

    def test_triangular_left_slope(self, calculator):
        """Test triangular function on left slope."""
        result = calculator.triangular(z=-1.0, left=-2.0, m=0.0, h=2.0)
        assert result == 0.5

    def test_triangular_right_slope(self, calculator):
        """Test triangular function on right slope."""
        result = calculator.triangular(z=1.0, left=-2.0, m=0.0, h=2.0)
        assert result == 0.5

    def test_sigmoid_center(self, calculator):
        """Test sigmoid function at center."""
        result = calculator.sigmoid(z=1.0, x0=1.0, k=2.0)
        assert result == 0.5

    def test_sigmoid_positive(self, calculator):
        """Test sigmoid function above center."""
        result = calculator.sigmoid(z=2.0, x0=1.0, k=2.0)
        assert result > 0.5
        assert result < 1.0

    def test_sigmoid_negative(self, calculator):
        """Test sigmoid function below center."""
        result = calculator.sigmoid(z=0.0, x0=1.0, k=2.0)
        assert result < 0.5
        assert result > 0.0

    def test_exponential_ramp_below_threshold(self, calculator):
        """Test exponential ramp below threshold."""
        result = calculator.exponential_ramp(z=0.0, tau=0.5, lam=1.5)
        assert result == 0.0

    def test_exponential_ramp_at_threshold(self, calculator):
        """Test exponential ramp at threshold."""
        result = calculator.exponential_ramp(z=0.5, tau=0.5, lam=1.5)
        assert result == 0.0

    def test_exponential_ramp_above_threshold(self, calculator):
        """Test exponential ramp above threshold."""
        result = calculator.exponential_ramp(z=1.5, tau=0.5, lam=1.5)
        assert result > 0.0
        assert result < 1.0

    def test_exponential_ramp_far_above_threshold(self, calculator):
        """Test exponential ramp far above threshold."""
        result = calculator.exponential_ramp(z=10.0, tau=0.5, lam=1.5)
        assert result > 0.99  # Should approach 1.0

    def test_gaussian_at_center(self, calculator):
        """Test gaussian at center."""
        result = calculator.gaussian(z=0.0, c=0.0, sigma=1.0)
        assert result == 1.0

    def test_gaussian_one_sigma(self, calculator):
        """Test gaussian at one sigma."""
        result = calculator.gaussian(z=1.0, c=0.0, sigma=1.0)
        expected = math.exp(-0.5)
        assert abs(result - expected) < 0.001

    def test_gaussian_two_sigma(self, calculator):
        """Test gaussian at two sigma."""
        result = calculator.gaussian(z=2.0, c=0.0, sigma=1.0)
        expected = math.exp(-2.0)
        assert abs(result - expected) < 0.001

    def test_gaussian_symmetry(self, calculator):
        """Test gaussian symmetry."""
        result_pos = calculator.gaussian(z=1.5, c=0.0, sigma=1.0)
        result_neg = calculator.gaussian(z=-1.5, c=0.0, sigma=1.0)
        assert abs(result_pos - result_neg) < 0.001  # Should be symmetric

    def test_calculate_triangular(self, calculator):
        """Test calculate dispatcher with triangular function."""
        membership_fn = {"type": "triangular", "params": {"l": -2.0, "m": 0.0, "h": 2.0}}
        result = calculator.calculate(z=0.0, membership_fn=membership_fn)
        assert result == 1.0

    def test_calculate_sigmoid(self, calculator):
        """Test calculate dispatcher with sigmoid function."""
        membership_fn = {"type": "sigmoid", "params": {"x0": 1.0, "k": 2.0}}
        result = calculator.calculate(z=1.0, membership_fn=membership_fn)
        assert result == 0.5

    def test_calculate_exponential_ramp(self, calculator):
        """Test calculate dispatcher with exponential_ramp function."""
        membership_fn = {
            "type": "exponential_ramp",
            "params": {"tau": 0.5, "lam": 1.5},
        }
        result = calculator.calculate(z=0.0, membership_fn=membership_fn)
        assert result == 0.0

    def test_calculate_gaussian(self, calculator):
        """Test calculate dispatcher with gaussian function."""
        membership_fn = {"type": "gaussian", "params": {"c": 0.0, "sigma": 1.0}}
        result = calculator.calculate(z=0.0, membership_fn=membership_fn)
        assert result == 1.0

    def test_calculate_unknown_type_raises(self, calculator):
        """Test calculate dispatcher with unknown function type."""
        membership_fn = {"type": "unknown", "params": {}}
        with pytest.raises(ValueError, match="Unknown membership function type"):
            calculator.calculate(z=0.0, membership_fn=membership_fn)
