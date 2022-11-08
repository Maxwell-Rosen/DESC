"""Script for creating plots in dudt2022."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from desc.basis import DoubleFourierSeries
from desc.compute import (
    compute_boozer_coords,
    compute_geometry,
    compute_quasisymmetry_error,
)
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, QuadratureGrid
from desc.transform import Transform

plt.rcParams["font.size"] = 14
dim = 11

# m=1, n=2 boundary Fourier coefficients
RBC = np.linspace(-0.1, 0, dim)
ZBS = np.linspace(-0.05, 0.05, dim)

# equilibrium
eq = Equilibrium.load("data/initial.h5")

grid = LinearGrid(M=6 * eq.M + 1, N=6 * eq.N + 1, NFP=eq.NFP, sym=False, rho=1.0)
qgrid = QuadratureGrid(M=3 * eq.M + 1, N=3 * eq.N + 1, L=3 * eq.L + 1, NFP=eq.NFP)
R_transform = Transform(qgrid, eq.R_basis, derivs=3)
Z_transform = Transform(qgrid, eq.Z_basis, derivs=3)
L_transform = Transform(qgrid, eq.L_basis, derivs=3)
B_transform = Transform(
    qgrid,
    DoubleFourierSeries(M=2 * eq.M, N=2 * eq.N, sym=eq.R_basis.sym, NFP=eq.NFP),
    derivs=0,
    build_pinv=True,
)
w_transform = Transform(
    qgrid,
    DoubleFourierSeries(M=2 * eq.M, N=2 * eq.N, sym=eq.Z_basis.sym, NFP=eq.NFP),
    derivs=1,
)

iota = eq.iota.copy()
iota.grid = qgrid


def qs_errors(eq):
    """Evaluate QS errors."""
    eq.surface.R_basis._create_idx()
    eq.surface.Z_basis._create_idx()

    data = compute_geometry(eq.R_lmn, eq.Z_lmn, R_transform, Z_transform)
    data = compute_boozer_coords(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        B_transform,
        w_transform,
        iota,
        data=data,
    )
    data = compute_quasisymmetry_error(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        helicity=(1, eq.NFP),
        data=data,
    )

    idx_00 = B_transform.basis.get_idx(M=0, N=0)
    idx_MN = np.where(
        B_transform.basis.modes[:, 1] / B_transform.basis.modes[:, 2] == 1
    )[0]
    idx = np.ones((B_transform.basis.num_modes,), bool)
    idx[idx_00] = False
    idx[idx_MN] = False
    R0 = np.mean(data["R"] * data["sqrt(g)"]) / np.mean(data["sqrt(g)"])
    B0 = np.mean(data["|B|"] * data["sqrt(g)"]) / np.mean(data["sqrt(g)"])

    f_B = np.sqrt(np.sum(data["|B|_mn"][idx] ** 2)) / np.sqrt(
        np.sum(data["|B|_mn"] ** 2)
    )
    f_C = (
        np.mean(np.abs(data["f_C"]) * data["sqrt(g)"])
        / np.mean(data["sqrt(g)"])
        / B0**3
    )
    f_T = (
        np.mean(np.abs(data["f_T"]) * data["sqrt(g)"])
        / np.mean(data["sqrt(g)"])
        * (R0**2 / B0**4)
    )

    return f_B, f_C, f_T


f_B = np.load("data/f_B.npy")
f_C = np.load("data/f_C.npy")
f_T = np.load("data/f_T.npy")

q_offset = 1e5

eq_initial = Equilibrium.load("data/initial.h5")
fB_initial, fC_initial, fT_initial = qs_errors(eq_initial)
qs_initial = [
    fB_initial,
    fC_initial,
    fT_initial,
    2.8363,
    1.15049716e-5 * 3.27712009 * q_offset,
    1.82336691e-5 * 3.27712009 * q_offset,
]

eq_stellopt = Equilibrium.load("data/stellopt.h5")
fB_stellopt, fC_stellopt, fT_stellopt = qs_errors(eq_stellopt)
qs_stellopt = [
    fB_stellopt,
    fC_stellopt,
    fT_stellopt,
    0.0158,
    2.14340292e-7 * 3.32096497 * q_offset,
    3.41165204e-7 * 3.32096497 * q_offset,
]
rbc_stellopt = np.load("data/rbc_stellopt.npy")
zbs_stellopt = np.load("data/zbs_stellopt.npy")

rbc_desc_fB_or1 = np.load("data/rbc_fB_or1.npy")
zbs_desc_fB_or1 = np.load("data/zbs_fB_or1.npy")

eq_fB_or2 = Equilibrium.load("data/eq_fB_or2.h5")
fB_fB_or2, fC_fB_or2, fT_fB_or2 = qs_errors(eq_fB_or2)
qs_fB_or2 = [
    fB_fB_or2,
    fC_fB_or2,
    fT_fB_or2,
    0.0208,
    2.78961246e-7 * 3.31954358 * q_offset,
    4.43670161e-7 * 3.31954358 * q_offset,
]
rbc_desc_fB_or2 = np.load("data/rbc_fB_or2.npy")
zbs_desc_fB_or2 = np.load("data/zbs_fB_or2.npy")

rbc_desc_fC_or1 = np.load("data/rbc_fC_or1.npy")
zbs_desc_fC_or1 = np.load("data/zbs_fC_or1.npy")

eq_fC_or2 = Equilibrium.load("data/eq_fC_or2.h5")
fB_fC_or2, fC_fC_or2, fT_fC_or2 = qs_errors(eq_fC_or2)
qs_fC_or2 = [
    fB_fC_or2,
    fC_fC_or2,
    fT_fC_or2,
    0.0173,
    2.34600482e-7 * 3.32042789 * q_offset,
    3.73470882e-7 * 3.32042789 * q_offset,
]
rbc_desc_fC_or2 = np.load("data/rbc_fC_or2.npy")
zbs_desc_fC_or2 = np.load("data/zbs_fC_or2.npy")

rbc_desc_fT_or1 = np.load("data/rbc_fT_or1.npy")
zbs_desc_fT_or1 = np.load("data/zbs_fT_or1.npy")

eq_fT_or2 = Equilibrium.load("data/eq_fT_or2.h5")
fB_fT_or2, fC_fT_or2, fT_fT_or2 = qs_errors(eq_fT_or2)
qs_fT_or2 = [
    fB_fT_or2,
    fC_fT_or2,
    fT_fT_or2,
    0.0030,
    3.48515364e-8 * 3.33237965 * q_offset,
    5.77730904e-8 * 3.33237965 * q_offset,
]
rbc_desc_fT_or2 = np.load("data/rbc_fT_or2.npy")
zbs_desc_fT_or2 = np.load("data/zbs_fT_or2.npy")

db = 1e-2
extent = [
    np.min(RBC) - db / 2,
    np.max(RBC) + db / 2,
    np.min(ZBS) - db / 2,
    np.max(ZBS) + db / 2,
]

blue = "#0504AA"
purple = "#9A0EEA"
pink = "#FC5A50"
green = "#15B01A"
yellow = "#FAC205"

# errors
fig, ax = plt.subplots(figsize=(7, 6), sharex=True, sharey=True)
labels = [
    r"$\hat{f}_{B}$",
    r"$\hat{f}_{C}$",
    r"$\hat{f}_{T}$",
    r"$\epsilon_{eff}$",
    r"$\hat{f}_{P} \times 10^{5}$",
    r"$\hat{f}_{Q} \times 10^{5}$",
]
x = np.arange(len(labels))
width = 0.18
bar_initial = ax.bar(x - 2 * width, qs_initial, width, color=blue, label="Initial")
bar_desc_fB = ax.bar(x - width, qs_fB_or2, width, color=purple, label=r"DESC $f_B$")
bar_desc_fC = ax.bar(x, qs_fC_or2, width, color=pink, label=r"DESC $f_C$")
bar_desc_fT = ax.bar(x + width, qs_fT_or2, width, color=green, label=r"DESC $f_T$")
bar_stellopt = ax.bar(x + 2 * width, qs_stellopt, width, color=yellow, label="STELLOPT")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale("log")
ax.set_xlim([-0.55, 5.55])
ax.set_ylim([1e-3, 1e1])
ax.set_ylabel("Normalized Error")
ax.legend(loc="upper left")
fig.tight_layout()
plt.savefig("data/errors.png")

# Boozer comparison
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(7, 10), sharex=True, sharey=True)
grid_plot = LinearGrid(M=100, N=100, NFP=eq.NFP, sym=False, endpoint=True)
xx = grid_plot.nodes[:, 2].reshape((grid_plot.M, grid_plot.N), order="F").squeeze()
yy = grid_plot.nodes[:, 1].reshape((grid_plot.M, grid_plot.N), order="F").squeeze()
B_transform = Transform(
    grid_plot,
    DoubleFourierSeries(M=2 * eq.M, N=2 * eq.N, sym=eq.R_basis.sym, NFP=eq.NFP),
)
data0 = eq.compute("|B|_mn", grid)
data0 = B_transform.transform(data0["|B|_mn"])
data0 = data0.reshape((grid_plot.M, grid_plot.N), order="F")
eq = Equilibrium.load("data/eq_fB_or2.h5")
data1 = eq.compute("|B|_mn", grid)
data1 = B_transform.transform(data1["|B|_mn"])
data1 = data1.reshape((grid_plot.M, grid_plot.N), order="F")
amin = min(np.nanmin(data0), np.nanmin(data1))
amax = max(np.nanmax(data0), np.nanmax(data1))
contourf_kwargs = {}
contourf_kwargs["norm"] = Normalize()
contourf_kwargs["levels"] = np.linspace(amin, amax, 100)
contourf_kwargs["cmap"] = "jet"
contourf_kwargs["extend"] = "both"
cax_kwargs = {"size": "5%", "pad": 0.05}
div0 = make_axes_locatable(ax0)
div1 = make_axes_locatable(ax1)
im0 = ax0.contourf(xx, yy, data0, **contourf_kwargs)
im1 = ax1.contourf(xx, yy, data1, **contourf_kwargs)
cax0 = div0.append_axes("right", **cax_kwargs)
cax1 = div1.append_axes("right", **cax_kwargs)
cbar0 = fig.colorbar(im0, cax=cax0)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar0.update_ticks()
cbar1.update_ticks()
ax1.set_xlabel(r"$\zeta_{Boozer}$")
ax0.set_ylabel(r"$\theta_{Boozer}$")
ax1.set_ylabel(r"$\theta_{Boozer}$")
ax0.set_title(r"Initial $|\mathbf{B}|~(T)$")
ax1.set_title(r"Optimized $|\mathbf{B}|~(T)$")
fig.tight_layout()
plt.savefig("data/Booz.png")

# f_B
fig, ax = plt.subplots(figsize=(7, 6), sharex=True, sharey=True)
div = make_axes_locatable(ax)
im = ax.imshow(
    f_B[0:11, 0:11], extent=extent, origin="lower", cmap="gray", norm=LogNorm()
)
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax.plot(rbc_stellopt, zbs_stellopt, "o-", c="#FAC205", lw=6, ms=12, label="STELLOPT")
ax.plot(
    rbc_desc_fB_or1,
    zbs_desc_fB_or1,
    "o--",
    c="#9A0EEA",
    lw=6,
    ms=12,
    label=r"DESC $f_B$ $1^{st}$-order",
)
ax.plot(
    rbc_desc_fB_or2,
    zbs_desc_fB_or2,
    "o:",
    c="#15B01A",
    lw=6,
    ms=12,
    label=r"DESC $f_B$ $2^{nd}$-order",
)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xticks(np.arange(extent[0] + db / 2, extent[1] + db / 2, step=5 * db))
ax.set_yticks(np.arange(extent[2] + db / 2, extent[3] + db / 2, step=5 * db))
ax.legend(loc="upper left")
ax.set_xlabel(r"$RBC(2,1)$")
ax.set_ylabel(r"$ZBS(2,1)$")
ax.set_title(r"$\hat{f}_{B}$")
fig.tight_layout()
plt.savefig("data/f_B.png")

# f_C
fig, ax = plt.subplots(figsize=(7, 6), sharex=True, sharey=True)
div = make_axes_locatable(ax)
im = ax.imshow(
    f_C[0:11, 0:11], extent=extent, origin="lower", cmap="gray", norm=LogNorm()
)
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax.plot(rbc_stellopt[-1], zbs_stellopt[-1], "o", c="#FAC205", ms=12, label="STELLOPT")
ax.plot(
    rbc_desc_fC_or1,
    zbs_desc_fC_or1,
    "o--",
    c="#9A0EEA",
    lw=6,
    ms=12,
    label=r"DESC $f_C$ $1^{st}$-order",
)
ax.plot(
    rbc_desc_fC_or2,
    zbs_desc_fC_or2,
    "o:",
    c="#15B01A",
    lw=6,
    ms=12,
    label=r"DESC $f_C$ $2^{nd}$-order",
)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xticks(np.arange(extent[0] + db / 2, extent[1] + db / 2, step=5 * db))
ax.set_yticks(np.arange(extent[2] + db / 2, extent[3] + db / 2, step=5 * db))
ax.legend(loc="upper left")
ax.set_xlabel(r"$RBC(2,1)$")
ax.set_ylabel(r"$ZBS(2,1)$")
ax.set_title(r"$\hat{f}_{C}$")
fig.tight_layout()
plt.savefig("data/f_C.png")

# f_T
fig, ax = plt.subplots(figsize=(7, 6), sharex=True, sharey=True)
div = make_axes_locatable(ax)
im = ax.imshow(
    f_T[0:11, 0:11], extent=extent, origin="lower", cmap="gray", norm=LogNorm()
)
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.update_ticks()
ax.plot(rbc_stellopt[-1], zbs_stellopt[-1], "o", c="#FAC205", ms=12, label="STELLOPT")
ax.plot(
    rbc_desc_fT_or1,
    zbs_desc_fT_or1,
    "o--",
    c="#9A0EEA",
    lw=6,
    ms=12,
    label=r"DESC $f_T$ $1^{st}$-order",
)
ax.plot(
    rbc_desc_fT_or2,
    zbs_desc_fT_or2,
    "o:",
    c="#15B01A",
    lw=6,
    ms=12,
    label=r"DESC $f_T$ $2^{nd}$-order",
)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xticks(np.arange(extent[0] + db / 2, extent[1] + db / 2, step=5 * db))
ax.set_yticks(np.arange(extent[2] + db / 2, extent[3] + db / 2, step=5 * db))
ax.legend(loc="upper left")
ax.set_xlabel(r"$RBC(2,1)$")
ax.set_ylabel(r"$ZBS(2,1)$")
ax.set_title(r"$\hat{f}_{T}$")
fig.tight_layout()
plt.savefig("data/f_T.png")

print("figures saved!")
