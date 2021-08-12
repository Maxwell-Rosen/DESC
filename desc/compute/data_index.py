"""data_index containts all of the quantities calculated by the compute functions.

label = (str) title of the quantity in LaTeX format
units = (str) units of the quantity in LaTeX format
fun = (str) function name in compute_funs.py that computes the quantity
dim = (int) dimension of the quantity: 0-D, 1-D, or 3-D
"""

data_index = {
    # toroidal flux
    "psi": {
        "label": "\\Psi \\ 2 \\pi",
        "units": "Wb",
        "fun": "compute_toroidal_flux",
        "dim": 1,
    },
    "psi_r": {
        "label": "\\Psi' \\ 2 \\pi",
        "units": "Wb",
        "fun": "compute_toroidal_flux",
        "dim": 1,
    },
    "psi_rr": {
        "label": "\\Psi'' \\ 2 \\pi",
        "units": "Wb",
        "fun": "compute_toroidal_flux",
        "dim": 1,
    },
    # pressure
    "p": {
        "label": "p",
        "units": "Pa",
        "fun": "compute_pressure",
        "dim": 1,
    },
    "p_r": {
        "label": "\\partial_{\\rho} p",
        "units": "Pa",
        "fun": "compute_pressure",
        "dim": 1,
    },
    # rotational transform
    "iota": {
        "label": "\\iota",
        "units": "",
        "fun": "compute_rotational_transform",
        "dim": 1,
    },
    "iota_r": {
        "label": "\\partial_{\\rho} \\iota",
        "units": "",
        "fun": "compute_rotational_transform",
        "dim": 1,
    },
    "iota_rr": {
        "label": "\\partial_{\\rho\\rho} \\iota",
        "units": "",
        "fun": "compute_rotational_transform",
        "dim": 1,
    },
    # R
    "R": {
        "label": "R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 0, 0]],
    },
    "R_r": {
        "label": "\\partial_{\\rho} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[1, 0, 0]],
    },
    "R_t": {
        "label": "\\partial_{\\theta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 1, 0]],
    },
    "R_z": {
        "label": "\\partial_{\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 0, 1]],
    },
    "R_rr": {
        "label": "\\partial_{\\rho\\rho} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[2, 0, 0]],
    },
    "R_tt": {
        "label": "\\partial_{\\theta\\theta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 2, 0]],
    },
    "R_zz": {
        "label": "\\partial_{\\zeta\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 0, 2]],
    },
    "R_rt": {
        "label": "\\partial_{\\rho\\theta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[1, 1, 0]],
    },
    "R_rz": {
        "label": "\\partial_{\\rho\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[1, 0, 1]],
    },
    "R_tz": {
        "label": "\\partial_{\\theta\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 1, 1]],
    },
    "R_rrr": {
        "label": "\\partial_{\\rho\\rho\\rho} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[3, 0, 0]],
    },
    "R_ttt": {
        "label": "\\partial_{\\theta\\theta\\theta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 3, 0]],
    },
    "R_zzz": {
        "label": "\\partial_{\\zeta\\zeta\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 0, 3]],
    },
    "R_rrt": {
        "label": "\\partial_{\\rho\\rho\\theta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[2, 1, 0]],
    },
    "R_rtt": {
        "label": "\\partial_{\\rho\\theta\\theta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[1, 2, 0]],
    },
    "R_rrz": {
        "label": "\\partial_{\\rho\\rho\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[2, 0, 1]],
    },
    "R_rzz": {
        "label": "\\partial_{\\rho\\zeta\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[1, 0, 2]],
    },
    "R_ttz": {
        "label": "\\partial_{\\theta\\theta\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 2, 1]],
    },
    "R_tzz": {
        "label": "\\partial_{\\theta\\zeta\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[0, 1, 2]],
    },
    "R_rtz": {
        "label": "\\partial_{\\rho\\theta\\zeta} R",
        "units": "m",
        "fun": "compute_R",
        "dim": 1,
        "R_derivs": [[1, 1, 1]],
    },
    # Z
    "Z": {
        "label": "Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 0, 0]],
    },
    "Z_r": {
        "label": "\\partial_{\\rho} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[1, 0, 0]],
    },
    "Z_t": {
        "label": "\\partial_{\\theta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 1, 0]],
    },
    "Z_z": {
        "label": "\\partial_{\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 0, 1]],
    },
    "Z_rr": {
        "label": "\\partial_{\\rho\\rho} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[2, 0, 0]],
    },
    "Z_tt": {
        "label": "\\partial_{\\theta\\theta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 2, 0]],
    },
    "Z_zz": {
        "label": "\\partial_{\\zeta\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 0, 2]],
    },
    "Z_rt": {
        "label": "\\partial_{\\rho\\theta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[1, 1, 0]],
    },
    "Z_rz": {
        "label": "\\partial_{\\rho\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[1, 0, 1]],
    },
    "Z_tz": {
        "label": "\\partial_{\\theta\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 1, 1]],
    },
    "Z_rrr": {
        "label": "\\partial_{\\rho\\rho\\rho} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[3, 0, 0]],
    },
    "Z_ttt": {
        "label": "\\partial_{\\theta\\theta\\theta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 3, 0]],
    },
    "Z_zzz": {
        "label": "\\partial_{\\zeta\\zeta\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 0, 3]],
    },
    "Z_rrt": {
        "label": "\\partial_{\\rho\\rho\\theta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[2, 1, 0]],
    },
    "Z_rtt": {
        "label": "\\partial_{\\rho\\theta\\theta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[1, 2, 0]],
    },
    "Z_rrz": {
        "label": "\\partial_{\\rho\\rho\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[2, 0, 1]],
    },
    "Z_rzz": {
        "label": "\\partial_{\\rho\\zeta\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[1, 0, 2]],
    },
    "Z_ttz": {
        "label": "\\partial_{\\theta\\theta\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 2, 1]],
    },
    "Z_tzz": {
        "label": "\\partial_{\\theta\\zeta\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[0, 1, 2]],
    },
    "Z_rtz": {
        "label": "\\partial_{\\rho\\theta\\zeta} Z",
        "units": "m",
        "fun": "compute_Z",
        "dim": 1,
        "Z_derivs": [[1, 1, 1]],
    },
    # lambda
    "lambda": {
        "label": "\\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 0, 0]],
    },
    "lambda_r": {
        "label": "\\partial_{\\rho} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[1, 0, 0]],
    },
    "lambda_t": {
        "label": "\\partial_{\\theta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 1, 0]],
    },
    "lambda_z": {
        "label": "\\partial_{\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 0, 1]],
    },
    "lambda_rr": {
        "label": "\\partial_{\\rho\\rho} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[2, 0, 0]],
    },
    "lambda_tt": {
        "label": "\\partial_{\\theta\\theta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 2, 0]],
    },
    "lambda_zz": {
        "label": "\\partial_{\\zeta\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 0, 2]],
    },
    "lambda_rt": {
        "label": "\\partial_{\\rho\\theta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[1, 1, 0]],
    },
    "lambda_rz": {
        "label": "\\partial_{\\rho\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[1, 0, 1]],
    },
    "lambda_tz": {
        "label": "\\partial_{\\theta\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 1, 1]],
    },
    "lambda_rrr": {
        "label": "\\partial_{\\rho\\rho\\rho} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[3, 0, 0]],
    },
    "lambda_ttt": {
        "label": "\\partial_{\\theta\\theta\\theta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 3, 0]],
    },
    "lambda_zzz": {
        "label": "\\partial_{\\zeta\\zeta\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 0, 3]],
    },
    "lambda_rrt": {
        "label": "\\partial_{\\rho\\rho\\theta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[2, 1, 0]],
    },
    "lambda_rtt": {
        "label": "\\partial_{\\rho\\theta\\theta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[1, 2, 0]],
    },
    "lambda_rrz": {
        "label": "\\partial_{\\rho\\rho\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[2, 0, 1]],
    },
    "lambda_rzz": {
        "label": "\\partial_{\\rho\\zeta\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[1, 0, 2]],
    },
    "lambda_ttz": {
        "label": "\\partial_{\\theta\\theta\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 2, 1]],
    },
    "lambda_tzz": {
        "label": "\\partial_{\\theta\\zeta\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[0, 1, 2]],
    },
    "lambda_rtz": {
        "label": "\\partial_{\\rho\\theta\\zeta} \\lambda",
        "units": "",
        "fun": "compute_lambda",
        "dim": 1,
        "L_derivs": [[1, 1, 1]],
    },
    # cartesian coordinates
    "X": {
        "label": "R \\cos{\\phi}",
        "units": "m",
        "fun": "compute_cartesian_coords",
        "dim": 1,
        "R_derivs": [[0, 0, 0]],
    },
    "Y": {
        "label": "R \\sin{\\phi}",
        "units": "m",
        "fun": "compute_cartesian_coords",
        "dim": 1,
        "R_derivs": [[0, 0, 0]],
    },
    # covariant basis
    "e_rho": {
        "label": "e_{\\rho}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 0, 0]],
        "Z_derivs": [[1, 0, 0]],
    },
    "e_theta": {
        "label": "e_{\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 1, 0]],
        "Z_derivs": [[0, 1, 0]],
    },
    "e_zeta": {
        "label": "e_{\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 0, 0], [0, 0, 1]],
        "Z_derivs": [[0, 0, 1]],
    },
    "e_rho_r": {
        "label": "e_{\\rho\\rho}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[2, 0, 0]],
        "Z_derivs": [[2, 0, 0]],
    },
    "e_rho_t": {
        "label": "e_{\\rho\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 1, 0]],
        "Z_derivs": [[1, 1, 0]],
    },
    "e_rho_z": {
        "label": "e_{\\rho\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 0, 1]],
        "Z_derivs": [[1, 0, 1]],
    },
    "e_theta_r": {
        "label": "e_{\\theta\\rho}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 1, 0]],
        "Z_derivs": [[1, 1, 0]],
    },
    "e_theta_t": {
        "label": "e_{\\theta\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 2, 0]],
        "Z_derivs": [[0, 2, 0]],
    },
    "e_theta_z": {
        "label": "e_{\\theta\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 1, 1]],
        "Z_derivs": [[0, 1, 1]],
    },
    "e_zeta_r": {
        "label": "e_{\\zeta\\rho}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 0, 0], [1, 0, 1]],
        "Z_derivs": [[1, 0, 1]],
    },
    "e_zeta_t": {
        "label": "e_{\\zeta\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 1, 0], [0, 1, 1]],
        "Z_derivs": [[0, 1, 1]],
    },
    "e_zeta_z": {
        "label": "e_{\\zeta\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 0, 1], [0, 0, 2]],
        "Z_derivs": [[0, 0, 2]],
    },
    "e_rho_rr": {
        "label": "e_{\\rho\\rho\\rho}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[3, 0, 0]],
        "Z_derivs": [[3, 0, 0]],
    },
    "e_rho_tt": {
        "label": "e_{\\rho\\theta\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 2, 0]],
        "Z_derivs": [[1, 2, 0]],
    },
    "e_rho_zz": {
        "label": "e_{\\rho\\zeta\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 0, 2]],
        "Z_derivs": [[1, 0, 2]],
    },
    "e_rho_rt": {
        "label": "e_{\\rho\\rho\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[2, 1, 0]],
        "Z_derivs": [[2, 1, 0]],
    },
    "e_rho_rz": {
        "label": "e_{\\rho\\rho\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[2, 1, 0]],
        "Z_derivs": [[2, 1, 0]],
    },
    "e_rho_tz": {
        "label": "e_{\\rho\\theta\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 1, 1]],
        "Z_derivs": [[1, 1, 1]],
    },
    "e_theta_rr": {
        "label": "e_{\\theta\\rho\\rho}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[2, 1, 0]],
        "Z_derivs": [[2, 1, 0]],
    },
    "e_theta_tt": {
        "label": "e_{\\theta\\theta\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 3, 0]],
        "Z_derivs": [[0, 3, 0]],
    },
    "e_theta_zz": {
        "label": "e_{\\theta\\zeta\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 1, 2]],
        "Z_derivs": [[0, 1, 2]],
    },
    "e_theta_rt": {
        "label": "e_{\\theta\\rho\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 2, 0]],
        "Z_derivs": [[1, 2, 0]],
    },
    "e_theta_rz": {
        "label": "e_{\\theta\\rho\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 1, 1]],
        "Z_derivs": [[1, 1, 1]],
    },
    "e_theta_tz": {
        "label": "e_{\\theta\\theta\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 2, 1]],
        "Z_derivs": [[0, 2, 1]],
    },
    "e_zeta_rr": {
        "label": "e_{\\zeta\\rho\\rho}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[2, 0, 0], [2, 0, 1]],
        "Z_derivs": [[2, 0, 1]],
    },
    "e_zeta_tt": {
        "label": "e_{\\zeta\\theta\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 2, 0], [0, 2, 1]],
        "Z_derivs": [[0, 2, 1]],
    },
    "e_zeta_zz": {
        "label": "e_{\\zeta\\zeta\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 0, 2], [0, 0, 3]],
        "Z_derivs": [[0, 0, 3]],
    },
    "e_zeta_rt": {
        "label": "e_{\\zeta\\rho\\theta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 1, 0], [1, 1, 1]],
        "Z_derivs": [[1, 1, 1]],
    },
    "e_zeta_rz": {
        "label": "e_{\\zeta\\rho\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[1, 0, 1], [1, 0, 2]],
        "Z_derivs": [[1, 0, 2]],
    },
    "e_zeta_tz": {
        "label": "e_{\\zeta\\theta\\zeta}",
        "units": "m",
        "fun": "compute_covariant_basis",
        "dim": 3,
        "R_derivs": [[0, 1, 1], [0, 1, 2]],
        "Z_derivs": [[0, 1, 2]],
    },
    # contravariant basis
    "e^rho": {
        "label": "e^{\\rho}",
        "units": "m^{-1}",
        "fun": "compute_contravariant_basis",
        "dim": 3,
        "R_derivs": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Z_derivs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    },
    "e^theta": {
        "label": "e^{\\theta}",
        "units": "m^{-1}",
        "fun": "compute_contravariant_basis",
        "dim": 3,
        "R_derivs": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Z_derivs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    },
    "e^zeta": {
        "label": "e^{\\zeta}",
        "units": "m^{-1}",
        "fun": "compute_contravariant_basis",
        "dim": 3,
        "R_derivs": [[0, 0, 0]],
        "Z_derivs": [],
    },
    # Jacobian
    "sqrt(g)": {
        "label": "\\sqrt{g}",
        "units": "m^3",
        "fun": "compute_jacobian",
        "dim": 1,
        "R_derivs": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Z_derivs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    },
    "sqrt(g)_r": {
        "label": "\\partial_{\\rho} \\sqrt{g}",
        "units": "m^3",
        "fun": "compute_jacobian",
        "dim": 1,
        "R_derivs": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
        ],
        "Z_derivs": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [1, 0, 1]],
    },
    "sqrt(g)_t": {
        "label": "\\partial_{\\theta} \\sqrt{g}",
        "units": "m^3",
        "fun": "compute_jacobian",
        "dim": 1,
        "R_derivs": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        "Z_derivs": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 2, 0], [1, 1, 0], [0, 1, 1]],
    },
    "sqrt(g)_z": {
        "label": "\\partial_{\\zeta} \\sqrt{g}",
        "units": "m^3",
        "fun": "compute_jacobian",
        "dim": 1,
        "R_derivs": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 2],
            [1, 0, 1],
            [0, 1, 1],
        ],
        "Z_derivs": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2], [1, 0, 1], [0, 1, 1]],
    },
    "sqrt(g)_rr": {
        "label": "\\partial_{\\rho\\rho} \\sqrt{g}",
        "units": "m^3",
        "fun": "compute_jacobian",
        "dim": 1,
        "R_derivs": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [3, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
        ],
        "Z_derivs": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [3, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
        ],
    },
    "sqrt(g)_tt": {
        "label": "\\partial_{\\theta\\theta} \\sqrt{g}",
        "units": "m^3",
        "fun": "compute_jacobian",
        "dim": 1,
        "R_derivs": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 3, 0],
            [1, 2, 0],
            [0, 2, 1],
        ],
        "Z_derivs": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 3, 0],
            [1, 2, 0],
            [0, 2, 1],
        ],
    },
    "sqrt(g)_zz": {
        "label": "\\partial_{\\zeta\\zeta} \\sqrt{g}",
        "units": "m^3",
        "fun": "compute_jacobian",
        "dim": 1,
        "R_derivs": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 2],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 3],
            [1, 0, 2],
            [0, 1, 2],
        ],
        "Z_derivs": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 2],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 3],
            [1, 0, 2],
            [0, 1, 2],
        ],
    },
    "sqrt(g)_tz": {
        "label": "\\partial_{\\theta\\zeta} \\sqrt{g}",
        "units": "m^3",
        "fun": "compute_jacobian",
        "dim": 1,
        "R_derivs": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 0],
            [0, 0, 2],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 2, 1],
            [0, 1, 2],
            [1, 1, 1],
        ],
        "Z_derivs": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 0],
            [0, 0, 2],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 2, 1],
            [0, 1, 2],
            [1, 1, 1],
        ],
    },
    # magnetic field
    "B": {
        "label": "B",
        "units": "T",
        "fun": "compute_contravariant_magnetic_field",
        "dim": 3,
        "R_derivs": [[0, 0, 0]],
        "Z_derivs": [[0, 0, 0]],
        "L_derivs": [[0, 0, 0]],
    },
    # volume
    "V": {
        "label": "V",
        "units": "m^3",
        "fun": "compute_volume",
        "dim": 0,
    },
    # energy
    "W": {
        "label": "W",
        "units": "J",
        "fun": "compute_energy",
        "dim": 0,
    },
    "W_B": {
        "label": "W_B",
        "units": "J",
        "fun": "compute_energy",
        "dim": 0,
    },
    "W_p": {
        "label": "W_p",
        "units": "J",
        "fun": "compute_energy",
        "dim": 0,
    },
}
