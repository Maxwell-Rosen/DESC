"""Tests for coils and coilsets."""

import shutil

import numpy as np
import pytest

from desc.coils import CoilSet, FourierPlanarCoil, FourierRZCoil, FourierXYZCoil
from desc.geometry import FourierRZCurve
from desc.grid import Grid, LinearGrid


class TestCoil:
    """Tests for singular coil objects."""

    @pytest.mark.unit
    def test_biot_savart(self):
        """Test biot-savart implementation against analytic formula."""
        R = 2
        y = 1
        I = 1
        By_true = 1e-7 * 2 * np.pi * R**2 * I / (y**2 + R**2) ** (3 / 2)
        B_true = np.array([0, By_true, 0])
        coil = FourierXYZCoil(I)
        coil.grid = LinearGrid(zeta=100, endpoint=True)
        assert coil.grid.num_nodes == 100
        B_approx = coil.compute_magnetic_field(
            Grid([[10, y, 0], [10, -y, 0]]), basis="xyz"
        )[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    @pytest.mark.unit
    def test_properties(self):
        """Test getting/setting attributes for Coil class."""
        current = 4.34
        coil = FourierPlanarCoil(current)
        assert coil.current == current
        new_current = 3.5
        coil.current = new_current
        assert coil.current == new_current


class TestCoilSet:
    """Tests for sets of multiple coils."""

    @pytest.mark.unit
    def test_linspaced_linear(self):
        """Field from straight solenoid."""
        R = 10
        z = np.linspace(0, 10, 10)
        I = 1
        Bz_true = np.sum(1e-7 * 2 * np.pi * R**2 * I / (z**2 + R**2) ** (3 / 2))
        B_true = np.array([0, 0, Bz_true])
        coil = FourierRZCoil(0.1)
        coils = CoilSet.linspaced_linear(
            coil, displacement=[0, 0, 10], n=10, endpoint=True
        )
        coils.current = I
        np.testing.assert_allclose(coils.current, I)
        coils.grid = 32
        assert coils.grid.N == 32
        B_approx = coils.compute_magnetic_field([0, 0, z[-1]], basis="xyz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    @pytest.mark.unit
    def test_linspaced_angular(self):
        """Field from uniform toroidal solenoid."""
        R = 10
        N = 50
        I = 1
        Bp_true = np.sum(1e-7 * 4 * np.pi * N * I / 2 / np.pi / R)
        B_true = np.array([0, Bp_true, 0])
        coil = FourierPlanarCoil()
        coil.current = I
        coils = CoilSet.linspaced_angular(coil, n=N)
        coils.grid = 32
        assert all([coil.grid.N == 32 for coil in coils])
        B_approx = coils.compute_magnetic_field([10, 0, 0], basis="rpz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    @pytest.mark.unit
    def test_from_symmetry(self):
        """Same toroidal solenoid field, but different construction."""
        R = 10
        N = 48
        I = 1
        Bp_true = np.sum(1e-7 * 4 * np.pi * N * I / 2 / np.pi / R)
        B_true = np.array([0, Bp_true, 0])
        coil = FourierPlanarCoil()
        coils = CoilSet.linspaced_angular(coil, angle=np.pi / 2, n=N // 4)
        coils = CoilSet.from_symmetry(coils, NFP=4)
        coils.grid = 32
        assert all([coil.grid.N == 32 for coil in coils])
        B_approx = coils.compute_magnetic_field([10, 0, 0], basis="rpz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

        # with stellarator symmetry
        NFP = 4
        coil = FourierXYZCoil()
        coil.rotate(angle=np.pi / N)
        coils = CoilSet.linspaced_angular(
            coil, I, [0, 0, 1], np.pi / NFP, N // NFP // 2
        )
        coils.grid = 32
        assert coils.grid.N == 32
        coils2 = CoilSet.from_symmetry(coils, NFP, True)
        B_approx = coils2.compute_magnetic_field([10, 0, 0], basis="rpz")[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    @pytest.mark.unit
    def test_properties(self):
        """Test getting/setting of CoilSet attributes."""
        coil = FourierPlanarCoil()
        coils = CoilSet.linspaced_linear(coil, n=4)
        coils.grid = np.array([[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(
            coils.compute_coordinates(),
            np.array(
                [
                    [12, 0, 0],
                    [12.5, 0, 0],
                    [13, 0, 0],
                    [13.5, 0, 0],
                ]
            ).reshape((4, 1, 3)),
        )
        np.testing.assert_allclose(coils.compute_curvature(), 1 / 2)
        np.testing.assert_allclose(coils.compute_torsion(), 0)
        TNB = coils.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz")
        T = [foo[0] for foo in TNB]
        N = [foo[1] for foo in TNB]
        B = [foo[2] for foo in TNB]
        np.testing.assert_allclose(
            T,
            np.array(
                [
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            N,
            np.array(
                [
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [-1, 0, 0],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            B,
            np.array(
                [
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        coils.grid = 32
        np.testing.assert_allclose(coils.compute_length(), 2 * 2 * np.pi)
        coils.translate([1, 1, 1])
        np.testing.assert_allclose(coils.compute_length(), 2 * 2 * np.pi)
        coils.flip([1, 0, 0])
        coils.grid = np.array([[0.0, 0.0, 0.0]])
        TNB = coils.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz")
        T = [foo[0] for foo in TNB]
        N = [foo[1] for foo in TNB]
        B = [foo[2] for foo in TNB]
        np.testing.assert_allclose(
            T,
            np.array(
                [
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            N,
            np.array(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            B,
            np.array(
                [
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )

    @pytest.mark.unit
    def test_dunder_methods(self):
        """Test methods for combining and calling CoilSet objects."""
        coil1 = FourierXYZCoil()
        coils1 = CoilSet.from_symmetry(coil1, NFP=4)
        coil2 = FourierPlanarCoil()
        coils2 = coils1 + [coil2]
        assert coils2[-1] is coil2
        coils2 = coils1 + CoilSet(coil2)
        assert coils2[-1] is coil2

        with pytest.raises(TypeError):
            _ = coils1 + FourierRZCurve()

        with pytest.raises(TypeError):
            coils1[-1] = FourierRZCurve()

        coils1[-1] = coil2
        assert coils1[-1] is coil2

        coils1.insert(-1, coil2)
        with pytest.raises(TypeError):
            coils1.insert(-1, FourierRZCurve())

        assert len(coils1) == 5

        assert coils1[-1] is coil2
        assert coils1[-2] is coil2

        s = coils1[-2:]
        assert s[-1] is coil2

        del coils1[-2]
        assert len(coils1) == 4
        assert coils1[-1] is coil2
        assert coils1[-2][0].__class__ is coil1.__class__


@pytest.mark.unit
def test_save_and_load_MAKEGRID_coils(tmpdir_factory):
    """Test loading in and saving CoilSets from MAKEGRID format files."""
    Ncoils = 22
    input_path = f"./tests/inputs/coils.MAKEGRID_format_{Ncoils}_coils"
    tmpdir = tmpdir_factory.mktemp("coil_files")
    tmp_path = tmpdir.join("coils.MAKEGRID_format_{Ncoils}_coils")
    shutil.copyfile(input_path, tmp_path)

    coilset = CoilSet.from_makegrid_coilfile(str(tmp_path))
    assert len(coilset) == Ncoils  # correct number of coils
    # TODO: add better tests for this?
    path = tmpdir.join("coils.MAKEGRID_format_desc")
    coilset.save_in_MAKEGRID_format(str(path))

    with open(tmp_path) as f:
        lines_orig = f.readlines()
    with open(path) as f:
        lines_new = f.readlines()
    for line_orig, line_new in zip(lines_orig, lines_new):
        assert line_orig == line_new


@pytest.mark.unit
def test_save_and_load_MAKEGRID_coils_Planar(tmpdir_factory):
    """Test saving and reloading CoilSet with PlanarCoil from MAKEGRID file."""
    tmpdir = tmpdir_factory.mktemp("coil_files")
    path = tmpdir.join("coils.MAKEGRID_format_planar_coil")

    # make a coilset with a single FourierPlanarCoil
    coilset = CoilSet(FourierPlanarCoil(r_n=np.array([2, 0]), modes=np.array([0, 1])))
    grid = LinearGrid(N=200, endpoint=True)
    coilset.save_in_MAKEGRID_format(str(path), grid=grid)

    coilset2 = CoilSet.from_makegrid_coilfile(str(path))

    # check values at saved points, ensure they match
    coords1 = coilset[0].compute_coordinates(grid=grid, basis="xyz")
    X1 = coords1[:, 0]
    Y1 = coords1[:, 1]
    Z1 = coords1[:, 2]

    coords2 = coilset2[0].compute_coordinates(grid=grid, basis="xyz")
    X2 = coords2[:, 0]
    Y2 = coords2[:, 1]
    Z2 = coords2[:, 2]

    np.testing.assert_allclose(coilset2[0].current, coilset[0].current)
    np.testing.assert_allclose(X1, X2)
    np.testing.assert_allclose(Y1, Y2)
    np.testing.assert_allclose(Z1, Z2, atol=9e-15)

    # check values at interpolated points, ensure they match closely
    grid = LinearGrid(N=50, endpoint=True)
    coords1 = coilset[0].compute_coordinates(grid=grid, basis="xyz")
    X1 = coords1[:, 0]
    Y1 = coords1[:, 1]
    Z1 = coords1[:, 2]

    coords2 = coilset2[0].compute_coordinates(grid=grid, basis="xyz")
    X2 = coords2[:, 0]
    Y2 = coords2[:, 1]
    Z2 = coords2[:, 2]

    np.testing.assert_allclose(coilset2[0].current, coilset[0].current)
    np.testing.assert_allclose(X1, X2, atol=1e-15)
    np.testing.assert_allclose(Y1, Y2, atol=1e-15)
    np.testing.assert_allclose(Z1, Z2, atol=1e-15)


@pytest.mark.unit
def test_save_MAKEGRID_coils_assert(tmpdir_factory):
    """Test saving CoilSet that with incompatible NFP throws an error."""
    Ncoils = 22
    input_path = f"./tests/inputs/coils.MAKEGRID_format_{Ncoils}_coils"
    tmpdir = tmpdir_factory.mktemp("coil_files")
    tmp_path = tmpdir.join("coils.MAKEGRID_format_{Ncoils}_coils")
    shutil.copyfile(input_path, tmp_path)

    coilset = CoilSet.from_makegrid_coilfile(str(tmp_path))
    assert len(coilset) == Ncoils  # correct number of coils
    path = tmpdir.join("coils.MAKEGRID_format_desc")
    with pytest.raises(AssertionError):
        coilset.save_in_MAKEGRID_format(str(path), NFP=3)


# TODO: add test where the CoilSet is not only XYZcoils


@pytest.mark.unit
def test_repr():
    """Test string representation of Coil objects."""
    coil = FourierRZCoil()
    assert "FourierRZCoil" in str(coil)
    assert "current=1" in str(coil)

    coils = CoilSet.linspaced_angular(coil, n=4)
    assert "CoilSet" in str(coils)
    assert "4 submembers" in str(coils)

    coils.name = "MyCoils"
    assert "MyCoils" in str(coils)
