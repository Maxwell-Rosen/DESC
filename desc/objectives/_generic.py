"""Generic objectives that don't belong anywhere else."""

from inspect import signature

import desc.compute as compute_funs
from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.compute import (
    arg_order,
    compute_boozer_magnetic_field,
    compute_rotational_transform,
    data_index,
)
from desc.compute.utils import compress
from desc.grid import LinearGrid, QuadratureGrid
from desc.transform import Transform
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class GenericObjective(_Objective):
    """A generic objective that can compute any quantity from the `data_index`.

    Parameters
    ----------
    f : str
        Name of the quantity to compute.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(Unknown)"
    _print_value_fmt = "Residual: {:10.3e} "

    def __init__(
        self,
        f,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="generic",
    ):

        self.f = f
        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )
        self._units = "(" + data_index[self.f]["units"] + ")"

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self.grid is None:
            self.grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)

        args = []
        self._dim_f = self.grid.num_nodes

        self.fun = getattr(compute_funs, data_index[self.f]["fun"])
        self.sig = signature(self.fun)
        self.inputs = {"data": None}

        for arg in self.sig.parameters.keys():
            if arg in arg_order:
                args.append(arg)
            elif arg == "R_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.R_basis,
                    derivs=data_index[self.f]["R_derivs"],
                    build=True,
                )
            elif arg == "Z_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.Z_basis,
                    derivs=data_index[self.f]["R_derivs"],
                    build=True,
                )
            elif arg == "L_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.L_basis,
                    derivs=data_index[self.f]["L_derivs"],
                    build=True,
                )
            elif arg == "B_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    DoubleFourierSeries(
                        M=2 * eq.M, N=2 * eq.N, sym=eq.R_basis.sym, NFP=eq.NFP
                    ),
                    derivs=0,
                    build_pinv=True,
                )
            elif arg == "w_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    DoubleFourierSeries(
                        M=2 * eq.M, N=2 * eq.N, sym=eq.Z_basis.sym, NFP=eq.NFP
                    ),
                    derivs=1,
                )
            elif arg == "pressure":
                self.inputs[arg] = eq.pressure.copy()
                self.inputs[arg].grid = self.grid
            elif arg == "iota":
                if eq.iota is not None:
                    self.inputs[arg] = eq.iota.copy()
                    self.inputs[arg].grid = self.grid
                else:
                    self.inputs[arg] = None
            elif arg == "current":
                if eq.current is not None:
                    self.inputs[arg] = eq.current.copy()
                    self.inputs[arg].grid = self.grid
                else:
                    self.inputs[arg] = None

        self._args = args
        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, **kwargs):
        """Compute the quantity.

        Parameters
        ----------
        args : list of ndarray
            Any of the arguments given in `arg_order`.

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        data = self.fun(**kwargs, **self.inputs)
        f = data[self.f] * self.grid.weights
        return self._shift_scale(f)


class TargetCurrent(_Objective):
    """Target toroidal current profile.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f == grid.num_rho
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f == grid.num_rho
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False
    _units = "(A)"
    _print_value_fmt = "Target current: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="toroidal current",
    ):

        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self.grid is None:
            self.grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )

        self._dim_f = self.grid.num_rho

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["current"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["current"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["current"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["I"] / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
        """Compute toroidal current.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        current : ndarray
            Toroidal current (A) through specified surfaces.

        """
        data = compute_boozer_magnetic_field(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
        )
        I = compress(self.grid, data["current"], surface_label="rho")
        w = compress(self.grid, self.grid.spacing[:, 0], surface_label="rho")
        return self._shift_scale(I) * w


class TargetIota(_Objective):
    """Targets a rotational transform profile.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f == grid.num_rho
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f == grid.num_rho
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.
    """

    _scalar = False
    _linear = False
    _units = "(n/a)"
    _print_value_fmt = "Target iota: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="target-iota",
    ):

        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.
        """
        if self.grid is None:
            self.grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )

        self._dim_f = self.grid.num_rho

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["iota"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["iota"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["iota"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
        """Compute rotational transform profile errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        iota : ndarray
            rotational transform on specified flux surfaces.
        """
        data = compute_rotational_transform(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
        )
        iota = compress(self.grid, data["iota"], surface_label="rho")
        w = compress(self.grid, self.grid.spacing[:, 0], surface_label="rho")
        return self._shift_scale(iota) * w
