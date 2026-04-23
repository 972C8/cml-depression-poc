"""Fuzzy membership functions for context evaluation.

This module implements triangular and trapezoidal membership functions
for fuzzy logic computations.
"""

import logging
from dataclasses import dataclass, field

from src.core.config import MembershipFunction

__all__ = [
    "LinguisticVariable",
    "MembershipCalculator",
]

logger = logging.getLogger(__name__)


class MembershipCalculator:
    """Calculator for fuzzy membership functions.

    Supports triangular and trapezoidal membership functions with
    deterministic computation.
    """

    def triangular(self, x: float, left: float, peak: float, right: float) -> float:
        """Calculate triangular membership value.

        Args:
            x: Input value to evaluate
            left: Left foot of triangle (mu=0)
            peak: Peak of triangle (mu=1)
            right: Right foot of triangle (mu=0)

        Returns:
            Membership degree in [0, 1]

        Raises:
            ValueError: If parameters are not properly ordered
        """
        if not (left <= peak <= right):
            raise ValueError(
                f"Invalid triangular parameters: left={left}, peak={peak}, right={right}. "
                f"Must satisfy left <= peak <= right"
            )

        # Handle peak at boundaries - must check before boundary conditions
        if x == peak:
            return 1.0

        if x <= left or x >= right:
            return 0.0
        elif left < x < peak:
            return (x - left) / (peak - left)
        else:  # peak < x < right
            return (right - x) / (right - peak)

    def trapezoidal(
        self,
        x: float,
        left: float,
        left_peak: float,
        right_peak: float,
        right: float,
    ) -> float:
        """Calculate trapezoidal membership value.

        Args:
            x: Input value to evaluate
            left: Left foot (mu=0)
            left_peak: Left shoulder (mu=1 starts)
            right_peak: Right shoulder (mu=1 ends)
            right: Right foot (mu=0)

        Returns:
            Membership degree in [0, 1]

        Raises:
            ValueError: If parameters are not properly ordered
        """
        if not (left <= left_peak <= right_peak <= right):
            raise ValueError(
                f"Invalid trapezoidal parameters: left={left}, left_peak={left_peak}, "
                f"right_peak={right_peak}, right={right}. "
                f"Must satisfy left <= left_peak <= right_peak <= right"
            )

        # Handle plateau at boundaries - must check before boundary conditions
        if left_peak <= x <= right_peak:
            return 1.0

        if x <= left or x >= right:
            return 0.0
        elif left < x < left_peak:
            return (x - left) / (left_peak - left)
        else:  # right_peak < x < right
            return (right - x) / (right - right_peak)

    def calculate(self, x: float, membership_fn: MembershipFunction) -> float:
        """Calculate membership value using the specified function.

        Args:
            x: Input value to evaluate
            membership_fn: MembershipFunction config with type and params

        Returns:
            Membership degree in [0, 1]

        Raises:
            ValueError: If function type is unknown
        """
        if membership_fn.type == "triangular":
            return self.triangular(x, *membership_fn.params)
        elif membership_fn.type == "trapezoidal":
            return self.trapezoidal(x, *membership_fn.params)
        else:
            raise ValueError(f"Unknown membership function type: {membership_fn.type}")


@dataclass
class LinguisticVariable:
    """A linguistic variable with multiple fuzzy sets.

    Example:
        people_in_room with sets: low, medium, high
    """

    name: str
    membership_fns: dict[str, MembershipFunction] = field(default_factory=dict)
    _calculator: MembershipCalculator = field(
        default_factory=MembershipCalculator, repr=False
    )

    def evaluate(self, value: float) -> dict[str, float]:
        """Evaluate value against all fuzzy sets.

        Args:
            value: Input value to evaluate

        Returns:
            Dict mapping fuzzy set name to membership degree
        """
        result = {}
        for set_name, membership_fn in self.membership_fns.items():
            result[set_name] = self._calculator.calculate(value, membership_fn)
        return result
