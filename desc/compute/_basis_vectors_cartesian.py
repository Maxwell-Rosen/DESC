from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import cross, safediv

@register_compute_fun(
    name="e_rho",
    label="\\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant Radial basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_r", "Z_r", "omega_r"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞ᵨ | ρ=0 }.
    data["e_rho"] = jnp.array([data["R_r"], data["omega_r"], data["Z_r"]]).T
    return data


@register_compute_fun(
    name="e_rho_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant Radial basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rr", "Z_rr", "omega_rr"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_r(params, transforms, profiles, data, **kwargs):
    # e_rho_r = a^i e_i, where the a^i are the components specified below and the
    # e_i are the basis vectors of the polar lab frame. omega_r e_2, -omega_r e_1,
    # 0 are the derivatives with respect to rho of e_1, e_2, e_3, respectively.
    data["e_rho_r"] = jnp.array(
        [
            data["R_rr"],
            data["omega_rr"],
            data["Z_rr"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt radial and radial"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rrr", "Z_rrr", "omega_rrr"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rr(params, transforms, profiles, data, **kwargs):
    data["e_rho_rr"] = jnp.array(
        [
            data["R_rrr"],
            data["omega_rrr"],
            data["Z_rrr"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rrr",
    label="\\partial_{\\rho \\rho \\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrrr",
        "Z_rrrr",
        "omega_rrrr",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rrr(params, transforms, profiles, data, **kwargs):
    data["e_rho_rrr"] = jnp.array(
        [
            data["R_rrrr"],
            data["omega_rrrr"],
            data["Z_rrrr"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rrt",
    label="\\partial_{\\rho \\rho \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
        " twice and poloidal once"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrrt",
        "Z_rrrt",
        "omega_rrrt",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rrt(params, transforms, profiles, data, **kwargs):
    data["e_rho_rrt"] = jnp.array(
        [
            data["R_rrrt"],
            data["omega_rrrt"],
            data["Z_rrrt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rrz",
    label="\\partial_{\\rho \\rho \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
        " twice and toroidal once"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrrz",
        "Z_rrrz",
        "omega_rrrz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rrz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rrz"] = jnp.array(
        [
            data["R_rrrz"],
            data["omega_rrrz"],
            data["Z_rrrz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt radial and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrt",
        "Z_rrt",
        "omega_rrt",
    ],
    aliases=["x_rrt", "x_rtr", "x_trr"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rt(params, transforms, profiles, data, **kwargs):
    data["e_rho_rt"] = jnp.array(
        [
            data["R_rrt"],
            data["omega_rrt"],
            data["Z_rrt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
        "once and poloidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrtt",
        "Z_rrtt",
        "omega_rrtt",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rtt(params, transforms, profiles, data, **kwargs):
    data["e_rho_rtt"] = jnp.array(
        [
            data["R_rrtt"],
            data["omega_rrtt"],
            data["Z_rrtt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial, poloidal,"
        " and toroidal coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrtz",
        "Z_rrtz",
        "omega_rrtz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rtz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rtz"] = jnp.array(
        [
            data["R_rrtz"],
            data["omega_rrtz"],
            data["Z_rrtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt radial and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrz",
        "Z_rrz",
        "omega_rrz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rz"] = jnp.array(
        [
            data["R_rrz"],
            data["omega_rrz"],
            data["Z_rrz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
        " once and toroidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrzz",
        "Z_rrzz",
        "omega_rrzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_rzz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rzz"] = jnp.array(
        [
            data["R_rrzz"] ,
            data["omega_rrzz"],
            data["Z_rrzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant Radial basis vector, derivative wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rt", "Z_rt", "omega_rt"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_t(params, transforms, profiles, data, **kwargs):
    data["e_rho_t"] = jnp.array(
        [
            data["R_rt"],
            data["omega_rt"],
            data["Z_rt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_tt",
    label="\\partial_{\\theta \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt poloidal and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rtt",
        "Z_rtt",
        "omega_rtt",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_tt(params, transforms, profiles, data, **kwargs):
    data["e_rho_tt"] = jnp.array(
        [
            data["R_rtt"],
            data["omega_rtt"],
            data["Z_rtt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt poloidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rtz",
        "Z_rtz",
        "omega_rtz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_tz(params, transforms, profiles, data, **kwargs):
    data["e_rho_tz"] = jnp.array(
        [
            data["R_rtz"],
            data["omega_rtz"],
            data["Z_rtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant Radial basis vector, derivative wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rz", "Z_rz", "omega_rz"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_z(params, transforms, profiles, data, **kwargs):
    data["e_rho_z"] = jnp.array(
        [
            data["R_rz"],
            data["omega_rz"],
            data["Z_rz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt toroidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rzz",
        "Z_rzz",
        "omega_rzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_rho_zz(params, transforms, profiles, data, **kwargs):
    data["e_rho_zz"] = jnp.array(
        [
            data["R_rzz"],
            data["omega_rzz"],
            data["Z_rzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta",
    label="\\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_t", "Z_t", "omega_t"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta(params, transforms, profiles, data, **kwargs):
    data["e_theta"] = jnp.array(
        [data["R_t"], data["omega_t"], data["Z_t"]]
    ).T
    return data


@register_compute_fun(
    name="e_theta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rt", "Z_rt", "omega_rt",],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_r(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { ∂ᵨ 𝐞_θ | ρ=0 }
    data["e_theta_r"] = jnp.array(
        [
            data["R_rt"],
            data["omega_rt"],
            data["Z_rt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt radial and radial"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrt",
        "Z_rrt",
        "omega_rrt",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rr(params, transforms, profiles, data, **kwargs):
    data["e_theta_rr"] = jnp.array(
        [
            data["R_rrt"],
            data["omega_rrt"],
            data["Z_rrt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rrr",
    label="\\partial_{\\rho \\rho \\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial coordinate"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrrt",
        "Z_rrrt",
        "omega_rrrt",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rrr(params, transforms, profiles, data, **kwargs):
    data["e_theta_rrr"] = jnp.array(
        [
            data["R_rrrt"],
            data["omega_rrrt"],
            data["Z_rrrt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rrt",
    label="\\partial_{\\rho \\rho \\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial coordinate"
        " twice and poloidal once"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrtt",
        "Z_rrtt",
        "omega_rrtt",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rrt(params, transforms, profiles, data, **kwargs):
    data["e_theta_rrt"] = jnp.array(
        [
            data["R_rrtt"],
            data["omega_rrtt"],
            data["Z_rrtt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rrz",
    label="\\partial_{\\rho \\rho \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial coordinate"
        " twice and toroidal once"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrtz",
        "Z_rrtz",
        "omega_rrtz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rrz(params, transforms, profiles, data, **kwargs):
    data["e_theta_rrz"] = jnp.array(
        [
            data["R_rrtz"] ,
            data["omega_rrtz"],
            data["Z_rrtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt radial and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rtt",
        "Z_rtt",
        "omega_rtt",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rt(params, transforms, profiles, data, **kwargs):
    data["e_theta_rt"] = jnp.array(
        [
            data["R_rtt"],
            data["omega_rtt"],
            data["Z_rtt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial coordinate"
        " once and poloidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rttt",
        "Z_rttt",
        "omega_rttt",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rtt(params, transforms, profiles, data, **kwargs):
    data["e_theta_rtt"] = jnp.array(
        [
            data["R_rttt"],
            data["omega_rttt"],
            data["Z_rttt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial, poloidal,"
        " and toroidal coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rttz",
        "Z_rttz",
        "omega_rttz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rtz(params, transforms, profiles, data, **kwargs):
    data["e_theta_rtz"] = jnp.array(
        [
            data["R_rttz"],
            data["omega_rttz"],
            data["Z_rttz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt radial and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rtz",
        "Z_rtz",
        "omega_rtz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rz(params, transforms, profiles, data, **kwargs):
    data["e_theta_rz"] = jnp.array(
        [
            data["R_rtz"],
            data["omega_rtz"] ,
            data["Z_rtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial coordinate"
        " once and toroidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rtzz",
        "Z_rtzz",
        "omega_rtzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_rzz(params, transforms, profiles, data, **kwargs):
    data["e_theta_rzz"] = jnp.array(
        [
            data["R_rtzz"] ,
            data["omega_rtzz"],
            data["Z_rtzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector, derivative wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_tt", "Z_tt", "omega_tt"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_t(params, transforms, profiles, data, **kwargs):
    data["e_theta_t"] = jnp.array(
        [
            data["R_tt"],
            data["omega_tt"],
            data["Z_tt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_tt",
    label="\\partial_{\\theta \\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt poloidal and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_ttt", "Z_ttt", "omega_ttt"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_tt(params, transforms, profiles, data, **kwargs):
    data["e_theta_tt"] = jnp.array(
        [
            data["R_ttt"],
            data["omega_ttt"],
            data["Z_ttt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt poloidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_ttz",
        "Z_ttz",
        "omega_ttz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_tz(params, transforms, profiles, data, **kwargs):
    data["e_theta_tz"] = jnp.array(
        [
            data["R_ttz"],
            data["omega_ttz"],
            data["Z_ttz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector, derivative wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_tz", "Z_tz", "omega_tz"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_z(params, transforms, profiles, data, **kwargs):
    data["e_theta_z"] = jnp.array(
        [
            data["R_tz"],
            data["omega_tz"],
            data["Z_tz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt toroidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_tzz",
        "Z_tzz",
        "omega_tzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_theta_zz(params, transforms, profiles, data, **kwargs):
    data["e_theta_zz"] = jnp.array(
        [
            data["R_tzz"],
            data["omega_tzz"],
            data["Z_tzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta",
    label="\\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant Toroidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_z", "Z_z", "omega_z"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta(params, transforms, profiles, data, **kwargs):
    data["e_zeta"] = jnp.array(
        [data["R_z"], 1 + data["omega_z"], data["Z_z"]]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant Toroidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rz", "Z_rz", "omega_rz",],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_r(params, transforms, profiles, data, **kwargs):
    data["e_zeta_r"] = jnp.array(
        [
            data["R_rz"],
            data["omega_rz"],
            data["Z_rz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt radial and radial"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrz",
        "Z_rrz",
        "omega_rrz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rr(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rr"] = jnp.array(
        [
            data["R_rrz"],
            data["omega_rrz"],
            data["Z_rrz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rrr",
    label="\\partial_{\\rho \\rho \\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, third derivative wrt radial coordinate"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrrz",
        "Z_rrrz",
        "omega_rrrz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rrr(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rrr"] = jnp.array(
        [
            data["R_rrrz"],
            data["omega_rrrz"],
            data["Z_rrrz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rrt",
    label="\\partial_{\\rho \\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, third derivative wrt radial coordinate"
        " twice and poloidal once"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrtz",
        "Z_rrtz",
        "omega_rrtz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rrt(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rrt"] = jnp.array(
        [
            data["R_rrtz"],
            data["omega_rrtz"],
            data["Z_rrtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rrz",
    label="\\partial_{\\rho \\rho \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, third derivative wrt radial coordinate"
        " twice and toroidal once"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rrzz",
        "Z_rrzz",
        "omega_rrzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rrz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rrz"] = jnp.array(
        [
            data["R_rrzz"],
            data["omega_rrzz"],
            data["Z_rrzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt radial and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rtz",
        "Z_rtz",
        "omega_rtz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rt(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rt"] = jnp.array(
        [
            data["R_rtz"],
            data["omega_rtz"],
            data["Z_rtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, third derivative wrt radial coordinate"
        " once and poloidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rttz",
        "Z_rttz",
        "omega_rttz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rtt(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rtt"] = jnp.array(
        [
            data["R_rttz"],
            data["omega_rttz"],
            data["Z_rttz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, third derivative wrt radial, poloidal,"
        " and toroidal coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rtzz",
        "Z_rtzz",
        "omega_rtzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rtz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rtz"] = jnp.array(
        [
            data["R_rtzz"],
            data["omega_rtzz"],
            data["Z_rtzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt radial and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rzz",
        "Z_rzz",
        "omega_rzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rz"] = jnp.array(
        [
            data["R_rzz"],
            data["omega_rzz"],
            data["Z_rzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, third derivative wrt radial coordinate"
        " once and toroidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_rzzz",
        "Z_rzzz",
        "omega_rzzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_rzz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rzz"] = jnp.array(
        [
            data["R_rzzz"],
            data["omega_rzzz"],
            data["Z_rzzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant Toroidal basis vector, derivative wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_tz", "Z_tz", "omega_tz",],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_t(params, transforms, profiles, data, **kwargs):
    data["e_zeta_t"] = jnp.array(
        [
            data["R_tz"],
            data["omega_tz"],
            data["Z_tz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_tt",
    label="\\partial_{\\theta \\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt poloidal and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_ttz",
        "Z_ttz",
        "omega_ttz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_tt(params, transforms, profiles, data, **kwargs):
    data["e_zeta_tt"] = jnp.array(
        [
            data["R_ttz"],
            data["omega_ttz"],
            data["Z_ttz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt poloidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R_tzz",
        "Z_tzz",
        "omega_tzz",
    ],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_tz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_tz"] = jnp.array(
        [
            data["R_tzz"],
            data["omega_tzz"],
            data["Z_tzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant Toroidal basis vector, derivative wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_zz", "Z_zz", "omega_zz"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_z(params, transforms, profiles, data, **kwargs):
    data["e_zeta_z"] = jnp.array(
        [
            data["R_zz"],
            data["omega_zz"],
            data["Z_zz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt toroidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_zzz", "Z_zzz", "omega_zzz"],
    parameterization="desc.equilibrium.equilibrium.PiecewiseEquilibriumCartesian",
)
def _e_sub_zeta_zz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_zz"] = jnp.array(
        [
            data["R_zzz"],
            data["omega_zzz"],
            data["Z_zzz"],
        ]
    ).T
    return data