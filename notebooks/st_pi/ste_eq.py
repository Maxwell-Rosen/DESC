from desc.transform import Transform
from desc.equilibrium import Equilibrium
from desc.basis import ChebyshevZernikeBasis, chebyshev_z
from desc.compute import compute
from desc.grid import LinearGrid, ConcentricGrid, QuadratureGrid, Grid
from desc.compute.utils import get_transforms
from desc.objectives import (
    FixEndCapLambda,
    FixEndCapR,
    FixEndCapZ,
    FixBoundaryR,
    FixBoundaryZ,
    FixPsi,
    FixPressure,
    FixIota,
    ForceBalance,
    ObjectiveFunction,
    CurrentDensity,
)
import numpy as np
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile
from desc.geometry import FourierRZToroidalSurface
from scipy.constants import mu_0


def chebygrid(N_grid):
    return np.concatenate(
        (
            [0],
            (-np.cos((2 * np.arange(N_grid) + 1) * np.pi / (2 * N_grid)) + 1)
            * np.pi
            / 2,
            [np.pi],
        )
    )


def grid_gen(L_grid, M_grid, N_grid, node_pattern="jacobi"):
    LMnodes = ConcentricGrid(L=L_grid, M=M_grid, N=0, node_pattern=node_pattern).nodes[
        :, :2
    ]
    Nnodes = chebygrid(N_grid)
    lm = np.tile(LMnodes, (Nnodes.size, 1))
    n = np.tile(Nnodes.reshape(-1, 1), (1, LMnodes.shape[0])).reshape(-1, 1)
    nodes = np.concatenate((lm, n), axis=1)
    return Grid(nodes)


def get_lm_mode(basis, coeff, zeta, L, M, func_zeta=chebyshev_z):
    modes = basis.modes
    lm = 0
    for i, (l, m, n) in enumerate(modes):
        if l == L and m == M:
            lm += func_zeta(zeta, n) * coeff[i]
    return lm


# 0.05,0.15
surf = FourierRZToroidalSurface(
    R_lmn=[10, 1, -0.0],
    modes_R=[[0, 0], [1, 0], [2, 1]],
    Z_lmn=[0, -1, 0.0],
    modes_Z=[[0, 0], [-1, 0], [-2, 1]],
    NFP=1,
    sym=False,
    mirror=True,
    length=np.pi,
)

p = PowerSeriesProfile(params=[0.00 / mu_0, -0.00 / mu_0], modes=[0, 2])
iota = PowerSeriesProfile(params=[0.0, 0])
eq = Equilibrium(
    surface=surf,
    L=3,
    M=2,
    N=4,
    mirror=True,
    pressure=p,
    iota=iota,
    sym=False,
    length=np.pi,
    ensure_nested=False,
)

constraints = (
    # FixEndCapLambda(0, eq=eq),
    # FixEndCapR(0, eq.eq),
    # FixEndCapZ(0, eq=eq),
    # FixEndCapLambda(2*np.pi, eq=eq),
    # FixEndCapR(2*np.pi, eq=eq),
    # FixEndCapZ(2*np.pi, eq=eq),
    FixBoundaryR(eq=eq),
    FixBoundaryZ(eq=eq),
    FixPressure(eq=eq),
    FixIota(eq=eq),
    FixPsi(eq=eq),
)
optimizer = Optimizer("lsq-exact")
grid = grid_gen(6, 4, 8)
# objectives = CurrentDensity(eq=eq, grid=grid)
objectives = ForceBalance(eq=eq, grid=grid)
obj = ObjectiveFunction(objectives=objectives)

eq.solve(
    objective=obj,
    constraints=constraints,
    optimizer=optimizer,
    ftol=1e-6,
    xtol=1e-16,
    gtol=1e-16,
    verbose=2,
)

eq.save("mirror_test.h5")

from desc.plotting import plot_section

fig, _ = plot_section(
    eq,
    "|F|",
    norm_F=True,
    log=True,
    grid=LinearGrid(
        rho=np.linspace(0, 1, 30),
        theta=np.linspace(0, 2 * np.pi, 30),
        zeta=np.linspace(0, np.pi, 6),
    ),
)
fig.savefig("error.png")