from desc.backend import jnp

from .data_index import register_compute_fun

@register_compute_fun(
    name="X",
    label="X = R \\cos{\\phi}",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "phi"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _X(params, transforms, profiles, data, **kwargs):
    data["X"] = data["R"] 
    return data


@register_compute_fun(
    name="X_r",
    label="\\partial_{\\rho} X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "phi", "phi_r"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _X_r(params, transforms, profiles, data, **kwargs):
    data["X_r"] = data["R_r"] 
    return data


@register_compute_fun(
    name="X_t",
    label="\\partial_{\\theta} X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate, derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_t", "phi", "phi_t"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _X_t(params, transforms, profiles, data, **kwargs):
    data["X_t"] = data["R_t"]
    return data


@register_compute_fun(
    name="X_z",
    label="\\partial_{\\zeta} X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate, derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_z", "phi", "phi_z"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _X_z(params, transforms, profiles, data, **kwargs):
    data["X_z"] = data["R_z"]
    return data


@register_compute_fun(
    name="Y",
    label="Y = R \\sin{\\phi}",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "phi"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _Y(params, transforms, profiles, data, **kwargs):
    data["Y"] = data["phi"] 
    return data


@register_compute_fun(
    name="Y_r",
    label="\\partial_{\\rho} Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi_r"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _Y_r(params, transforms, profiles, data, **kwargs):
    data["Y_r"] = data["phi_r"] 
    return data


@register_compute_fun(
    name="Y_t",
    label="\\partial_{\\theta} Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate, derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi_t"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _Y_t(params, transforms, profiles, data, **kwargs):
    data["Y_t"] =  data["phi_t"] 
    return data


@register_compute_fun(
    name="Y_z",
    label="\\partial_{\\zeta} Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate, derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi_z"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _Y_z(params, transforms, profiles, data, **kwargs):
    data["Y_z"] = data["phi_z"]
    return data