"""Biomarker membership function calculator."""

import logging
import math

logger = logging.getLogger(__name__)


class BiomarkerMembershipCalculator:
    """Calculator for biomarker membership functions."""

    def __init__(self):
        """Initialize calculator."""
        self._logger = logging.getLogger(__name__)

    def calculate(self, z: float, membership_fn: dict) -> float:
        """Calculate membership value from z-score.

        Args:
            z: Z-score value
            membership_fn: Membership function configuration with keys:
                - type: str (triangular, sigmoid, exponential_ramp, gaussian)
                - params: dict with function-specific parameters

        Returns:
            Membership value in [0, 1]

        Raises:
            ValueError: If function type is unknown

        """
        fn_type = membership_fn.get("type")
        params = membership_fn.get("params", {})

        if fn_type == "triangular":
            return self.triangular(z, left=params["l"], m=params["m"], h=params["h"])
        elif fn_type == "sigmoid":
            return self.sigmoid(z, x0=params["x0"], k=params["k"])
        elif fn_type == "exponential_ramp":
            return self.exponential_ramp(z, tau=params["tau"], lam=params["lam"])
        elif fn_type == "gaussian":
            return self.gaussian(z, c=params["c"], sigma=params["sigma"])
        else:
            msg = f"Unknown membership function type: {fn_type}"
            self._logger.error(msg)
            raise ValueError(msg)

    def triangular(self, z: float, left: float, m: float, h: float) -> float:
        """Calculate triangular membership value.

        Args:
            z: Z-score value
            left: Left boundary
            m: Peak (center)
            h: Right boundary

        Returns:
            Membership value in [0, 1]

        """
        if z <= left or z >= h:
            return 0.0
        if z <= m:
            return (z - left) / (m - left)
        return (h - z) / (h - m)

    def sigmoid(self, z: float, x0: float, k: float) -> float:
        """Calculate sigmoid membership value.

        Args:
            z: Z-score value
            x0: Center point
            k: Steepness

        Returns:
            Membership value in [0, 1]

        """
        return 1.0 / (1.0 + math.exp(-k * (z - x0)))

    def exponential_ramp(self, z: float, tau: float, lam: float) -> float:
        """Calculate exponential ramp membership value.

        Args:
            z: Z-score value
            tau: Threshold
            lam: Lambda (rate)

        Returns:
            Membership value in [0, 1]

        """
        if z <= tau:
            return 0.0
        return 1.0 - math.exp(-lam * (z - tau))

    def gaussian(self, z: float, c: float, sigma: float) -> float:
        """Calculate gaussian membership value.

        Args:
            z: Z-score value
            c: Center
            sigma: Width (standard deviation)

        Returns:
            Membership value in [0, 1]

        """
        return math.exp(-((z - c) ** 2) / (2.0 * sigma**2))
