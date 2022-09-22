import os
import numpy as np
from netCDF4 import Dataset
import pytest

from .utils import area_difference, compute_coords
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid
from desc.__main__ import main


def test_compute_geometry(DSHAPE, DSHAPE_current):
    """Test computation of plasma geometric values."""

    def test(stellarator):
        # VMEC values
        file = Dataset(str(stellarator["vmec_nc_path"]), mode="r")
        V_vmec = float(file.variables["volume_p"][-1])
        R0_vmec = float(file.variables["Rmajor_p"][-1])
        a_vmec = float(file.variables["Aminor_p"][-1])
        ar_vmec = float(file.variables["aspect"][-1])
        file.close()

        # DESC values
        eq = EquilibriaFamily.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        data = eq.compute("R0/a")
        V_desc = data["V"]
        R0_desc = data["R0"]
        a_desc = data["a"]
        ar_desc = data["R0/a"]

        assert abs(V_vmec - V_desc) < 5e-3
        assert abs(R0_vmec - R0_desc) < 5e-3
        assert abs(a_vmec - a_desc) < 5e-3
        assert abs(ar_vmec - ar_desc) < 5e-3

    test(DSHAPE)
    test(DSHAPE_current)


@pytest.mark.slow
def test_compute_theta_coords(SOLOVEV):
    """Test root finding for theta(theta*,lambda(theta))."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute("lambda", Grid(nodes, sort=False))
    flux_coords = nodes.copy()
    flux_coords[:, 1] += coords["lambda"]

    geom_coords = eq.compute_theta_coords(flux_coords)
    geom_coords = np.array(geom_coords)

    # catch difference between 0 and 2*pi
    if geom_coords[0, 1] > np.pi:  # theta[0] = 0
        geom_coords[0, 1] = geom_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, geom_coords, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_compute_flux_coords(SOLOVEV):
    """Test root finding for (rho,theta,zeta) from (R,phi,Z)."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute("R", Grid(nodes, sort=False))
    real_coords = np.vstack([coords["R"].flatten(), zeta, coords["Z"].flatten()]).T

    flux_coords = eq.compute_flux_coords(real_coords)
    flux_coords = np.array(flux_coords)

    # catch difference between 0 and 2*pi
    if flux_coords[0, 1] > np.pi:  # theta[0] = 0
        flux_coords[0, 1] = flux_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, flux_coords, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_to_sfl(SOLOVEV):
    """Test converting an equilibrium to straight field line coordinates."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    Rr1, Zr1, Rv1, Zv1 = compute_coords(eq)
    Rr2, Zr2, Rv2, Zv2 = compute_coords(eq.to_sfl())
    rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)

    np.testing.assert_allclose(rho_err, 0, atol=2.5e-5)
    np.testing.assert_allclose(theta_err, 0, atol=1e-7)


@pytest.mark.slow
def test_continuation_resolution(tmpdir_factory):
    """Test that stepping resolution in continuation method works correctly."""
    input_path = ".//tests//inputs//res_test"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("res_test_out.h5")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    with pytest.warns(UserWarning):
        main(args)


def test_grid_resolution_warning(SOLOVEV):
    """Test that a warning is thrown if grid resolution is too low."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    eqN = eq.copy()
    eqN.change_resolution(N=1, N_grid=0)
    with pytest.warns(Warning):
        eqN.solve(ftol=1e-2, maxiter=2)
    eqM = eq.copy()
    eqM.change_resolution(M=eq.M, M_grid=eq.M - 1)
    with pytest.warns(Warning):
        eqM.solve(ftol=1e-2, maxiter=2)
    eqL = eq.copy()
    eqL.change_resolution(L=eq.L, L_grid=eq.L - 1)
    with pytest.warns(Warning):
        eqL.solve(ftol=1e-2, maxiter=2)


def test_eq_change_grid_resolution(SOLOVEV):
    """Test changing equilibrium grid resolution."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    eq.change_resolution(L_grid=10, M_grid=10, N_grid=10)
    assert eq.L_grid == 10
    assert eq.M_grid == 10
    assert eq.N_grid == 10


def test_resolution():
    """Test changing equilibrium spectral resolution."""
    eq1 = Equilibrium(L=5, M=6, N=7, L_grid=8, M_grid=9, N_grid=10)
    eq2 = Equilibrium()

    assert eq1.resolution() != eq2.resolution()
    eq2.change_resolution(**eq1.resolution())
    assert eq1.resolution() == eq2.resolution()

    eq1.L = 2
    eq1.M = 3
    eq1.N = 4
    eq1.NFP = 5
    assert eq1.R_basis.L == 2
    assert eq1.R_basis.M == 3
    assert eq1.R_basis.N == 4
    assert eq1.R_basis.NFP == 5


def test_poincare_solve_not_implemented():
    """Test that solving with fixed poincare section doesn't work yet."""
    inputs = {
        "L": 4,
        "M": 2,
        "N": 2,
        "NFP": 3,
        "sym": False,
        "spectral_indexing": "ansi",
        "axis": np.array([[0, 10, 0]]),
        "pressure": np.array([[0, 10], [2, 5]]),
        "iota": np.array([[0, 1], [2, 3]]),
        "surface": np.array([[0, 0, 0, 10, 0], [1, 1, 0, 1, 1]]),
    }

    eq = Equilibrium(**inputs)
    np.testing.assert_allclose(
        eq.Rb_lmn, [10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    with pytest.raises(NotImplementedError):
        eq.solve()
