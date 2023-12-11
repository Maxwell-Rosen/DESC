"""Tests for transforming from spectral coefficients to real space values."""

import numpy as np
import pytest

from desc.basis import ChebyshevZernikeBasis
from desc.grid import ConcentricGrid
from desc.transform import Transform


class TestTransform:
    """Test the Transform class."""

    @pytest.mark.mirror_unit
    def test_volume_chebyshev_zernike(self):
        """Tests transform of Chebyshev-Zernike basis in a toroidal volume."""
        grid = ConcentricGrid(L=2, M=2, N=2)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=2, sym=None)
        transf = Transform(grid, basis, method = "direct1")
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates
        z_shift = z/np.pi - 1
        correct_vals = (
            2 * r  * np.cos(t) * z_shift
            - 0.5 * r * np.sin(t) *(2 * z_shift**2 - 1)
            + 1
        )
        idx_0 = np.where((basis.modes == [1, 1, 1]).all(axis=1))[0]
        idx_1 = np.where((basis.modes == [1, -1, 2]).all(axis=1))[0]
        idx_2 = np.where((basis.modes == [0, 0, 0]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        c[idx_1] = -0.5
        c[idx_2] = 1
        values = transf.transform(c, 0, 0, 0)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_first_derivatives_chebyshev_zernike(self):
        """Test CZ Basis, differentiating chebyz once from the first term."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,1])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        correct_vals = 2 * r  * np.cos(t) * 1 / np.pi
        idx_0 = np.where((basis.modes == [1, 1, 1]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 1)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_first_derivatives_second_chebyshev_zernike(self):
        """Test CZ transform differentiating 2nd cheby polynomial once."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,1])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates
        z_shift = z/np.pi - 1
        correct_vals = 2 * r  * np.cos(t) * 4*z_shift / np.pi
        idx_0 = np.where((basis.modes == [1, 1, 2]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 1)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_first_derivatives_third_chebyshev_zernike(self):
        """Test CZ transform differentiating 2nd cheby polynomial once."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,1])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates
        z_shift = z/np.pi - 1
        correct_vals = 2 * r  * np.cos(t) * (12*z_shift**2 - 3) / np.pi
        idx_0 = np.where((basis.modes == [1, 1, 3]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 1)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_second_derivatives_third_chebyshev_zernike(self):
        """Test CZ transform differentiating 2nd cheby polynomial once."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,2])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates
        z_shift = z/np.pi - 1
        correct_vals = (2 * r  * np.cos(t) * (24*z_shift**1)) / np.pi**2
        idx_0 = np.where((basis.modes == [1, 1, 3]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 2)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_third_derivatives_third_chebyshev_zernike(self):
        """Test CZ transform differentiating 2nd cheby polynomial once."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,3])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        correct_vals = (2 * r  * np.cos(t) * (24)) / np.pi**3
        idx_0 = np.where((basis.modes == [1, 1, 3]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 3)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_third_derivatives_second_chebyshev_zernike(self):
        """Test CZ transform differentiating 2nd cheby polynomial once."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,3])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        correct_vals = (2 * r  * np.cos(t) * (0)) / np.pi**3
        idx_0 = np.where((basis.modes == [1, 1, 2]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 3)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_third_derivatives_zeroth_chebyshev_zernike(self):
        """Test CZ transform differentiating 2nd cheby polynomial once."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,3])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        correct_vals = (2 * r  * np.cos(t) * (0)) / np.pi**3
        idx_0 = np.where((basis.modes == [1, 1, 0]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 3)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_second_derivatives_zeroth_chebyshev_zernike(self):
        """Test CZ transform differentiating 2nd cheby polynomial once."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,2])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        correct_vals = (2 * r  * np.cos(t) * (0)) / np.pi**2
        idx_0 = np.where((basis.modes == [1, 1, 0]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 2)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.mirror_unit
    def test_fourth_derivatives_fourth_chebyshev_zernike(self):
        """Test CZ transform differentiating 2nd cheby polynomial once."""
        grid = ConcentricGrid(L=2, M=2, N=8)
        basis = ChebyshevZernikeBasis(L=1, M=2, N=5, sym=None)
        transf = Transform(grid, basis, method = "direct1", derivs = [0,0,4])
        r = grid.nodes[:, 0]  # rho coordinates
        t = grid.nodes[:, 1]  # theta coordinates
        correct_vals = (2 * r  * np.cos(t) * (192)) / np.pi**4
        idx_0 = np.where((basis.modes == [1, 1, 4]).all(axis=1))[0]
        c = np.zeros((basis.modes.shape[0],))
        #Coeffieicints of each term in the correct expansion
        c[idx_0] = 2
        values = transf.transform(c, 0, 0, 4)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
