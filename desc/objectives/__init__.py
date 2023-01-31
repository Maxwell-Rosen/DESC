"""Classes defining objectives for equilibrium and optimization."""

from ._equilibrium import (
    CurrentDensity,
    Energy,
    ForceBalance,
    HelicalForceBalance,
    RadialForceBalance,
)
from ._generic import GenericObjective, RotationalTransform, ToroidalCurrent
from ._geometry import (
    AspectRatio,
    Elongation,
    MeanCurvature,
    PrincipalCurvature,
    Volume,
)
from ._qs import QuasisymmetryBoozer, QuasisymmetryTripleProduct, QuasisymmetryTwoTerm
from ._stability import MagneticWell, MercierStability
from ._wrappers import WrappedEquilibriumObjective
from .linear_objectives import (
    FixAtomicNumber,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambdaGauge,
    FixPressure,
    FixPsi,
)
from ._auglagrangian import AugLagrangianLS, AugLagrangian
from .objective_funs import ObjectiveFunction
from .utils import get_equilibrium_objective, get_fixed_boundary_constraints
