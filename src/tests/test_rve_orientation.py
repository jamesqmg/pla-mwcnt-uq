"""Tests for src/uq/rve_orientation.py -- RED phase (fail before implementation).

Directional Young's modulus E(theta) from a homogenized stiffness tensor, for the
Fig-6 raster-orientation story: rotate the load direction relative to the fiber
axis and read the modulus, via the exact compliance-tensor contraction
1/E(n) = S_ijkl n_i n_j n_k n_l. This reuses the aligned-fiber C (no diagonal-fiber
geometry needed).

Correctness anchors: (1) an isotropic C must give E(n) = E for EVERY direction
(this catches any Voigt<->tensor factor error); (2) with a diagonal compliance
diag(1/Et, 1/Et, 1/Ea, ...), a pure-axis direction picks off exactly that axis's
modulus, so E along z = Ea and E along x = Et exactly.

Run:  python -m pytest src/tests/test_rve_orientation.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from skfem.models.elasticity import lame_parameters

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.uq.rve_orientation import (  # noqa: E402
    youngs_modulus_in_direction,
    youngs_modulus_vs_angle,
)


def _isotropic_voigt_stiffness(E, nu):
    lam, mu = lame_parameters(E, nu)
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] = lam + 2 * mu
    for k in range(3, 6):
        C[k, k] = mu
    return C


class TestDirectionalModulusIsotropic:
    def test_isotropic_is_direction_independent(self):
        C = _isotropic_voigt_stiffness(3.5, 0.3)
        directions = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [2, -1, 3],
        ]
        for d in directions:
            assert youngs_modulus_in_direction(C, d) == pytest.approx(3.5, rel=1e-9)


class TestDirectionalModulusAligned:
    def _C(self, Ea=200.0, Et=5.0):
        # diagonal compliance: pure-axis directions pick off exactly one modulus
        S = np.diag([1 / Et, 1 / Et, 1 / Ea, 1.0, 1.0, 1.0])
        return np.linalg.inv(S)

    def test_axial_direction_gives_axial_modulus(self):
        C = self._C()
        assert youngs_modulus_in_direction(C, [0, 0, 1]) == pytest.approx(200.0, rel=1e-6)

    def test_transverse_direction_gives_transverse_modulus(self):
        C = self._C()
        assert youngs_modulus_in_direction(C, [1, 0, 0]) == pytest.approx(5.0, rel=1e-6)
        assert youngs_modulus_in_direction(C, [0, 1, 0]) == pytest.approx(5.0, rel=1e-6)

    def test_direction_normalized_internally(self):
        C = self._C()
        assert youngs_modulus_in_direction(C, [0, 0, 5]) == pytest.approx(200.0, rel=1e-6)


class TestYoungsModulusVsAngle:
    def _C(self, Ea=200.0, Et=5.0):
        S = np.diag([1 / Et, 1 / Et, 1 / Ea, 1.0, 1.0, 1.0])
        return np.linalg.inv(S)

    def test_endpoints_axial_to_transverse(self):
        C = self._C()
        angles = np.array([0.0, np.pi / 2])
        e = youngs_modulus_vs_angle(C, angles, axial_axis=2, transverse_axis=0)
        assert e[0] == pytest.approx(200.0, rel=1e-6)   # 0 deg = along fiber axis
        assert e[1] == pytest.approx(5.0, rel=1e-6)     # 90 deg = transverse

    def test_shape(self):
        C = self._C()
        angles = np.linspace(0, np.pi / 2, 10)
        e = youngs_modulus_vs_angle(C, angles)
        assert e.shape == (10,)

    def test_axial_is_stiffest_and_drops_off_axis(self):
        # E(theta) for an aligned-fiber material is NOT monotonic: the fiber axis
        # is the global maximum and the modulus falls sharply off-axis, commonly
        # dipping below the transverse value at intermediate angles (shear
        # coupling) -- a real, honest feature of the raster-orientation curve.
        C = self._C()
        angles = np.linspace(0, np.pi / 2, 7)
        e = youngs_modulus_vs_angle(C, angles, axial_axis=2, transverse_axis=0)
        assert e[0] == pytest.approx(max(e))   # fiber axis is stiffest
        assert e[1] < e[0]                     # drops off-axis
        assert e.min() <= e[-1] + 1e-9         # off-axis dip reaches at/below transverse
