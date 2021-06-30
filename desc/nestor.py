#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:15:05 2021

Neumann Solver for Toroidal Systems

@author: Jonathan Schilling (jonathan.schilling@ipp.mpg.de)
"""

import os
import sys

import numpy as np
from desc.backend import put
from desc.utils import Index
from desc.magnetic_fields import SplineMagneticField
from netCDF4 import Dataset

mu0 = 4.0e-7*np.pi

def def_ncdim(ncfile, size):
    dimname = "dim_%05d"%(size,)
    ncfile.createDimension(dimname, size)
    return dimname


def copy_vector_periods(vec, zetas):
    """Copies a vector into each field period by rotation

    Parameters
    ----------
    vec : ndarray, shape(3,...)
        vector(s) to rotate
    zetas : ndarray
        angles to rotate by (eg start of each field period)

    Returns
    -------
    vec : ndarray, shape(3,...,nzeta)
        vector(s) repeated and rotated by angle zeta
    """
    if vec.shape[0] == 3:
        x,y,z = vec
    else:
        x,y = vec
    shp = x.shape
    xx = x.reshape((*shp, 1)) * np.cos(zetas) - y.reshape((*shp, 1))*np.sin(zetas)
    yy = y.reshape((*shp, 1)) * np.cos(zetas) + x.reshape((*shp, 1))*np.sin(zetas)
    if vec.shape[0] == 3:
        zz = np.broadcast_to(z.reshape((*shp,1)),(*shp, zetas.size))
        return np.array((xx,yy,zz))
    return np.array((xx,yy))



    #eval surface geometry
def evalSurfaceGeometry_vmec(xm, xn, mnmax, ntheta, nzeta, ntheta_sym, nfp, rmnc, zmns, rmns=None, zmnc=None, sym=False):


    # integer mode number arrays
    ixm=np.array(np.round(xm), dtype=int)
    ixn=np.array(np.round(np.divide(xn, nfp)), dtype=int)        
    # Fourier mode sorting array
    mIdx = ixm
    # reverse toroidal mode numbers, since VMEC kernel (mu-nv) is reversed in n    
    nIdx = np.where(ixn<=0, -ixn, nzeta-ixn)
    

    # input arrays for FFTs
    Rmn   = np.zeros([ntheta, nzeta], dtype=np.complex128) # for R
    mRmn  = np.zeros([ntheta, nzeta], dtype=np.complex128) # for R_t
    nRmn  = np.zeros([ntheta, nzeta], dtype=np.complex128) # for R_z
    mmRmn = np.zeros([ntheta, nzeta], dtype=np.complex128) # for R_tt
    mnRmn = np.zeros([ntheta, nzeta], dtype=np.complex128) # for R_tz
    nnRmn = np.zeros([ntheta, nzeta], dtype=np.complex128) # for R_zz
    Zmn   = np.zeros([ntheta, nzeta], dtype=np.complex128) # for Z
    mZmn  = np.zeros([ntheta, nzeta], dtype=np.complex128) # for Z_t
    nZmn  = np.zeros([ntheta, nzeta], dtype=np.complex128) # for Z_z
    mmZmn = np.zeros([ntheta, nzeta], dtype=np.complex128) # for Z_tt
    mnZmn = np.zeros([ntheta, nzeta], dtype=np.complex128) # for Z_tz
    nnZmn = np.zeros([ntheta, nzeta], dtype=np.complex128) # for Z_zz

    # multiply with mode numbers to get tangential derivatives
    Rmn  = put(  Rmn,Index[mIdx, nIdx], rmnc)
    mRmn = put( mRmn,Index[mIdx, nIdx], -   xm*rmnc)
    nRmn = put( nRmn,Index[mIdx, nIdx],     xn*rmnc)
    mmRmn= put(mmRmn,Index[mIdx, nIdx],  -xm*xm*rmnc)
    mnRmn= put(mnRmn,Index[mIdx, nIdx],   xm*xn*rmnc)
    nnRmn= put(nnRmn,Index[mIdx, nIdx],  -xn*xn*rmnc)
    Zmn  = put(  Zmn,Index[mIdx, nIdx],  -      zmns*1j)
    mZmn = put( mZmn,Index[mIdx, nIdx],      xm*zmns*1j)
    nZmn = put( nZmn,Index[mIdx, nIdx],  -   xn*zmns*1j)
    mmZmn= put(mmZmn,Index[mIdx, nIdx],   xm*xm*zmns*1j)
    mnZmn= put(mnZmn,Index[mIdx, nIdx],  -xm*xn*zmns*1j)
    nnZmn= put(nnZmn,Index[mIdx, nIdx],   xn*xn*zmns*1j)
    # TODO: if lasym, must also include corresponding terms above!

    R_2d             = (np.fft.ifft2(  Rmn)*ntheta*nzeta).real
    R_t_2d      = (np.fft.ifft2( mRmn)*ntheta*nzeta).imag
    R_z_2d       = (np.fft.ifft2( nRmn)*ntheta*nzeta).imag
    R_tt_2d    = (np.fft.ifft2(mmRmn)*ntheta*nzeta).real
    R_tz_2d = (np.fft.ifft2(mnRmn)*ntheta*nzeta).real
    R_zz_2d     = (np.fft.ifft2(nnRmn)*ntheta*nzeta).real
    Z_2d             = (np.fft.ifft2(  Zmn)*ntheta*nzeta).real
    Z_t_2d      = (np.fft.ifft2( mZmn)*ntheta*nzeta).imag
    Z_z_2d       = (np.fft.ifft2( nZmn)*ntheta*nzeta).imag
    Z_tt_2d    = (np.fft.ifft2(mmZmn)*ntheta*nzeta).real
    Z_tz_2d = (np.fft.ifft2(mnZmn)*ntheta*nzeta).real
    Z_zz_2d     = (np.fft.ifft2(nnZmn)*ntheta*nzeta).real

    coords = {}
    # vectorize arrays, since most operations to follow act on all grid points anyway
    coords["R"] = R_2d.flatten()
    coords["Z"] = Z_2d.flatten()
    if sym:
        coords["R_sym"]         = R_2d            [:ntheta_sym,:].flatten()
        coords["Z_sym"]         = Z_2d            [:ntheta_sym,:].flatten()
        coords["R_t"]      = R_t_2d     [:ntheta_sym,:].flatten()
        coords["R_z"]       = R_z_2d      [:ntheta_sym,:].flatten()
        coords["R_tt"]    = R_tt_2d   [:ntheta_sym,:].flatten()
        coords["R_tz"] = R_tz_2d[:ntheta_sym,:].flatten()
        coords["R_zz"]     = R_zz_2d    [:ntheta_sym,:].flatten()
        coords["Z_t"]      = Z_t_2d     [:ntheta_sym,:].flatten()
        coords["Z_z"]       = Z_z_2d      [:ntheta_sym,:].flatten()
        coords["Z_tt"]    = Z_tt_2d   [:ntheta_sym,:].flatten()
        coords["Z_tz"] = Z_tz_2d[:ntheta_sym,:].flatten()
        coords["Z_zz"]     = Z_zz_2d    [:ntheta_sym,:].flatten()
    else:
        coords["R_sym"]         = coords["R"]
        coords["Z_sym"]         = coords["Z"]
        coords["R_t"]      = R_t_2d.flatten()
        coords["R_z"]       = R_z_2d.flatten()
        coords["R_tt"]    = R_tt_2d.flatten()
        coords["R_tz"] = R_tz_2d.flatten()
        coords["R_zz"]     = R_zz_2d.flatten()
        coords["Z_t"]      = Z_t_2d.flatten()
        coords["Z_z"]       = Z_z_2d.flatten()
        coords["Z_tt"]    = Z_tt_2d.flatten()
        coords["Z_tz"] = Z_tz_2d.flatten()
        coords["Z_zz"]     = Z_zz_2d.flatten()

    phi = np.linspace(0,2*np.pi,nzeta, endpoint=False)/nfp            
    coords["phi_sym"] = np.broadcast_to(phi, (ntheta_sym, nzeta)).flatten()


    coords["X"] = (R_2d * np.cos(phi)).flatten()
    coords["Y"] = (R_2d * np.sin(phi)).flatten()     
    return coords


def compute_normal(coords, signgs):
    normal = {}
    normal["R_n"]   =  signgs * (coords["R_sym"] * coords["Z_t"])
    normal["phi_n"] =  signgs * (coords["R_t"] * coords["Z_z"]
                                              - coords["R_z"] * coords["Z_t"])
    normal["Z_n"]   = -signgs * (coords["R_sym"] * coords["R_t"])
    return normal

def compute_jacobian(coords, normal, nfp):

    jacobian = {}        

    # a, b, c in NESTOR article: dot-products of first-order derivatives of surface
    jacobian["g_tt"] = (coords["R_t"] * coords["R_t"]
                        + coords["Z_t"] * coords["Z_t"])
    jacobian["g_tz"] = (coords["R_t"] * coords["R_z"]
                        + coords["Z_t"] * coords["Z_z"])/nfp
    jacobian["g_zz"] = (coords["R_z"]  * coords["R_z"]
                        + coords["Z_z"]  * coords["Z_z"]
                        + coords["R_sym"] * coords["R_sym"])/nfp**2
    
    # A, B and C in NESTOR article: surface normal dotted with second-order derivative of surface (?)
    jacobian["a_tt"]   = 0.5 * (normal["R_n"] * coords["R_tt"]
                                + normal["Z_n"] * coords["Z_tt"])
    jacobian["a_tz"] = (normal["R_n"] * coords["R_tz"]
                        + normal["phi_n"] * coords["R_t"]
                        + normal["Z_n"] * coords["Z_tz"])/nfp
    jacobian["a_zz"]   = (normal["phi_n"] * coords["R_z"] +
                          0.5*(normal["R_n"] * (coords["R_zz"] - coords["R_sym"])
                               + normal["Z_n"] * coords["Z_zz"]) )/nfp**2

    return jacobian


# TODO: vectorize this over multiple coils
def biot_savart(eval_pts, coil_pts, current):
    """Biot-Savart law following [1]

    Parameters
    ----------
    eval_pts : array-like shape(3,n)
        evaluation points in cartesian coordinates
    coil_pts : array-like shape(3,m)
        points in cartesian space defining coil
    current : float
        current through the coil

    Returns
    -------
    B : ndarray, shape(3,k)
        magnetic field in cartesian components at specified points

    [1] Hanson & Hirshman, "Compact expressions for the Biot-Savart fields of a filamentary segment" (2002)
    """
    dvec = np.diff(coil_pts, axis=1)
    L = np.linalg.norm(dvec, axis=0)

    Ri_vec = eval_pts[:, :,np.newaxis] - coil_pts[:,np.newaxis,:-1]
    Ri = np.linalg.norm(Ri_vec, axis=0)
    Rf = np.linalg.norm(eval_pts[:, :, np.newaxis] - coil_pts[:,np.newaxis,1:], axis=0)
    Ri_p_Rf = Ri + Rf

    # 1.0e-7 == mu0/(4 pi)
    Bmag = 1.0e-7 * current * 2.0 * Ri_p_Rf / ( Ri * Rf * (Ri_p_Rf*Ri_p_Rf - L*L) )

    # cross product of L*hat(eps)==dvec with Ri_vec, scaled by Bmag
    vec = np.cross(dvec, Ri_vec, axis=0)
    return np.sum(Bmag * vec, axis=-1)


# model net toroidal plasma current as filament along the magnetic axis
# and add its magnetic field on the LCFS to the MGRID magnetic field
def modelNetToroidalCurrent(raxis, phiaxis, zaxis, current, R_sym, phi_sym, Z_sym, zeta_fp):

    # TODO: we can simplify this by evaluating the field directly in cylindrical coordinates
    # copy 1 field period around to make full torus
    xyz = np.array([raxis*np.cos(phiaxis),
                    raxis*np.sin(phiaxis),
                    zaxis])
    xpts = np.moveaxis(copy_vector_periods(xyz, zeta_fp), -1,1).reshape((3,-1))
    # first point == last point        
    xpts = np.hstack([xpts[:,-1:], xpts])

    eval_pts = np.array([R_sym*np.cos(phi_sym),
                         R_sym*np.sin(phi_sym),
                         Z_sym])


    B = biot_savart(eval_pts, xpts, current)

    # add B^X and B^Y to MGRID magnetic field; need to convert to cylindrical components first
    return np.array([B[0]*np.cos(phi_sym) + B[1]*np.sin(phi_sym),
                     B[1]*np.cos(phi_sym) - B[0]*np.sin(phi_sym),
                     B[2]])


# Neumann Solver for Toroidal Systems
class Nestor:

    # error flag from/to VMEC
    ier_flag = None

    # skip counter --> only do full NESTOR calculation every nvacskip iterations
    ivacskip = None

    # 0,1,2; depending on initialization status of NESTOR in VMEC
    ivac = None

    # number of field periods
    nfp = None

    # number of toroidal Fourier harmonics in geometry input
    ntor = None

    # number of poloidal Fourier harmonics in geometry input
    mpol = None

    # number of toroidal grid points; has to match mgrid file!
    nzeta = None

    # number of poloidal grid points
    ntheta = None

    # total number of Fourier coefficients in geometry input
    mnmax = None

    # poloidal mode numbers m of geometry input
    xm = None

    # toroidal mode numbers n*nfp of geometry input
    xn = None

    # Fourier coefficients for R*cos(m theta - n zeta) of geometry input
    rmnc = None

    # Fourier coefficients for Z*sin(m theta - n zeta) of geometry input
    zmns = None

    # net poloidal current; only used for comparison
    rbtor = None

    # net toroial current in A*mu0; used for filament model along magnetic axis
    ctor = None

    # flag to indicate non-stellarator-symmetry mode
    lasym = None

    # sign of Jacobian; needed for surface normal vector sign
    signgs = None

    # coil currents for scaling mgrid file
    extcur = None

    # toroidal Fourier coefficients for magnetic axis: R*cos(n zeta)
    raxis_nestor = None

    # toroidal Fourier coefficients for magnetic axis: Z*sin(n zeta)
    zaxis_nestor = None

    # normalization factor for surface integrals;
    # essentially 1/(ntheta*nzeta) with 1/2 at the ends in the poloidal direction
    wint = None

    # bvec from previous iteration to be used when skipping full NESTOR calculation
    bvecsav = None

    # amat from previous iteration to be used when skipping full NESTOR calculation
    amatsav = None

    # poloidal current (?) from previous iteration; has to be carried over for use in VMEC
    bsubvvac = None

    # MGridFile object holding the external magnetic field to interpolate
    mgrid = None

    def __init__(self, vacinFilename, mgrid):
        self.vacin = Dataset(vacinFilename, "r")

        self.ier_flag        = int(self.vacin['ier_flag'][()])


        self.ivacskip        = int(self.vacin['ivacskip'][()])
        self.ivac            = int(self.vacin['ivac'][()])
        self.nfp             = int(self.vacin['nfp'][()])
        self.ntor            = int(self.vacin['ntor'][()])
        self.mpol            = int(self.vacin['mpol'][()])
        self.nzeta           = int(self.vacin['nzeta'][()])
        self.ntheta          = int(self.vacin['ntheta'][()])
        self.rbtor           = self.vacin['rbtor'][()]
        self.ctor            = self.vacin['ctor'][()]
        self.lasym           = (self.vacin['lasym__logical__'][()] != 0)
        self.signgs          = self.vacin['signgs'][()]

        self.raxis_nestor    = self.vacin['raxis_nestor'][()]
        self.zaxis_nestor    = self.vacin['zaxis_nestor'][()]
        self.wint            = np.array(self.vacin['wint'][()])
        self.bvecsav         = self.vacin['bvecsav'][()]
        self.amatsav         = self.vacin['amatsav'][()]
        self.bsubvvac        = self.vacin['bsubvvac'][()]
        # self.vacin.close()

        extcur          = self.vacin['extcur'][()]        
        folder = os.getcwd()
        mgridFilename = os.path.join(folder, mgrid)
        self.ext_field = SplineMagneticField.from_mgrid(mgridFilename, extcur)

    # pre-computable quantities and arrays
    def precompute(self):
        self.mf = self.mpol+1
        self.nf = self.ntor

        if self.nzeta == 1:
            self.nfp_eff = 64
        else:
            self.nfp_eff = self.nfp

        # toroidal angles for starting points of toroidal modules
        self.zeta_fp = 2.0*np.pi/self.nfp_eff * np.arange(self.nfp_eff)
        
        # tanu, tanv
        epstan = 2.22e-16
        bigno = 1.0e50 # allows proper comparison against implementation used in VMEC
        #bigno = np.inf # allows proper plotting

        self.tanu = 2.0*np.tan( np.pi*np.arange(2*self.ntheta)/self.ntheta )
        # mask explicit singularities at tan(pi/2), tan(3/2 pi)
        self.tanu = np.where( (np.arange(2*self.ntheta)/self.ntheta-0.5)%1 < epstan, bigno, self.tanu)

        if self.nzeta == 1:
            # Tokamak: need nfp_eff toroidal grid points
            argv = np.arange(self.nfp_eff)/self.nfp_eff
        else:
            # Stellarator: need nzeta toroidal grdi points
            argv = np.arange(self.nzeta)/self.nzeta
            
        self.tanv = 2.0*np.tan( np.pi*argv )
        # mask explicit singularities at tan(pi/2)
        self.tanv = np.where( (argv-0.5)%1 < epstan , bigno, self.tanv)

        cmn = np.zeros([self.mf+self.nf+1, self.mf+1, self.nf+1])
        for m in range(self.mf+1):
            for n in range(self.nf+1):
                jmn = m+n
                imn = m-n
                kmn = abs(imn)
                smn = (jmn+kmn)/2
                f1 = 1
                f2 = 1
                f3 = 1
                for i in range(1, kmn+1):
                    f1 *= (smn-(i-1))
                    f2 *= i
                for l in range(kmn, jmn+1, 2):
                    cmn[l,m,n] = f1/(f2*f3)*((-1)**((l-imn)/2))
                    f1 *= (jmn+l+2)*(jmn-l)/4
                    f2 *= (l+2+kmn)/2
                    f3 *= (l+2-kmn)/2

        # toroidal extent of one module
        dPhi_per = 2.0*np.pi/self.nfp                    
        # cmns from cmn
        self.cmns = np.zeros([self.mf+self.nf+1, self.mf+1, self.nf+1])
        for m in range(1, self.mf+1):
            for n in range(1, self.nf+1):
                self.cmns[:,m,n] = 0.5*dPhi_per*(cmn[:,m,n] + cmn[:, m-1, n] + cmn[:, m, n-1] + cmn[:, m-1, n-1])
        self.cmns[:,1:self.mf+1,0] = 0.5 * dPhi_per * (cmn[:,1:self.mf+1,0] + cmn[:,0:self.mf,0])
        self.cmns[:,0,1:self.nf+1] = 0.5 * dPhi_per * (cmn[:,0,1:self.nf+1] + cmn[:,0,0:self.nf])
        self.cmns[:,0,0] = 0.5 * dPhi_per * (cmn[:,0,0] + cmn[:,0,0])


        self.ntheta_stellsym = self.ntheta//2 + 1
        self.nzeta_stellsym = self.nzeta//2 + 1

    # evaluate MGRID on grid over flux surface
    def interpolateMGridFile(self, R, Z, phi):
        grid = np.array([R,phi,Z]).T
        B = self.ext_field.compute_magnetic_field(grid)

        return B.T


    def compute_T_S(self, jacobian):
        a = jacobian["g_tt"]
        b = jacobian["g_tz"]
        c = jacobian["g_zz"]
        ap = a + 2*b + c
        am = a - 2*b + c
        cma = c - a

        sqrt_a = np.sqrt(a)
        sqrt_c = np.sqrt(c)
        sqrt_ap = np.sqrt(ap)
        sqrt_am = np.sqrt(am)

        delt1u  = ap*am  - cma*cma
        azp1u  = jacobian["a_tt"]  + jacobian["a_tz"]  + jacobian["a_zz"]
        azm1u  = jacobian["a_tt"]  - jacobian["a_tz"]  + jacobian["a_zz"]
        cma11u  = jacobian["a_zz"]  - jacobian["a_tt"]
        r1p  = (azp1u*(delt1u - cma*cma)/ap - azm1u*ap + 2.0*cma11u*cma)/delt1u
        r1m  = (azm1u*(delt1u - cma*cma)/am - azp1u*am + 2.0*cma11u*cma)/delt1u
        r0p  = (-azp1u*am*cma/ap - azm1u*cma + 2.0*cma11u*am)/delt1u
        r0m  = (-azm1u*ap*cma/am - azp1u*cma + 2.0*cma11u*ap)/delt1u
        ra1p = azp1u/ap
        ra1m = azm1u/am

        # compute T^{\pm}_l, S^{\pm}_l

        num_four = self.mf + self.nf + 1
            
        # storage for all T^{\pm}_l, S^{\pm}_l
        T_p_l = np.zeros([num_four, self.ntheta_stellsym*self.nzeta]) # T^{+}_l
        T_m_l = np.zeros([num_four, self.ntheta_stellsym*self.nzeta]) # T^{-}_l
        S_p_l = np.zeros([num_four, self.ntheta_stellsym*self.nzeta]) # S^{+}_l
        S_m_l = np.zeros([num_four, self.ntheta_stellsym*self.nzeta]) # S^{-}_l

        # T^{\pm}_0
        T_p_l[0, :] = 1.0/sqrt_ap*np.log((sqrt_ap*2*sqrt_c + ap + cma)/(sqrt_ap*2*sqrt_a - ap + cma))
        T_m_l[0, :] = 1.0/sqrt_am*np.log((sqrt_am*2*sqrt_c + am + cma)/(sqrt_am*2*sqrt_a - am + cma))

        # S^{\pm}_0
        S_p_l[0, :] = ra1p * T_p_l[0, :] - (r1p + r0p)/(2*sqrt_c) + (r0p - r1p)/(2*sqrt_a)
        S_m_l[0, :] = ra1m * T_m_l[0, :] - (r1m + r0m)/(2*sqrt_c) + (r0m - r1m)/(2*sqrt_a)

        # now use recurrence relation for l > 0
        for l in range(1, self.mf+self.nf+1):

            # compute T^{\pm}_l
            if l > 1:
                T_p_l[l, :] = ((2*sqrt_c + (-1)**l * 2*sqrt_a) - (2.0*l - 1.0)*cma*T_p_l[l-1, :] - (l-1)*am*T_p_l[l-2, :])/(ap*l)
                T_m_l[l, :] = ((2*sqrt_c + (-1)**l * 2*sqrt_a) - (2.0*l - 1.0)*cma*T_m_l[l-1, :] - (l-1)*ap*T_m_l[l-2, :])/(am*l)
            else:
                T_p_l[l, :] = ((2*sqrt_c + (-1)**l * 2*sqrt_a) - (2.0*l - 1.0)*cma*T_p_l[l-1, :])/(ap*l)
                T_m_l[l, :] = ((2*sqrt_c + (-1)**l * 2*sqrt_a) - (2.0*l - 1.0)*cma*T_m_l[l-1, :])/(am*l)

            # compute S^{\pm}_l based on T^{\pm}_l and T^{\pm}_{l-1}
            S_p_l[l, :] = (r1p*l + ra1p)*T_p_l[l, :] + r0p*l*T_p_l[l-1, :] - (r1p + r0p)/(2*sqrt_c) + (-1)**l * (r0p - r1p)/(2*sqrt_a)
            S_m_l[l, :] = (r1m*l + ra1m)*T_m_l[l, :] + r0m*l*T_m_l[l-1, :] - (r1m + r0m)/(2*sqrt_c) + (-1)**l * (r0m - r1m)/(2*sqrt_a)

        return T_p_l, T_m_l, S_p_l, S_m_l
        
    def analyticalIntegrals(self, jacobian, normal, T_p_l, T_m_l, S_p_l, S_m_l, B_field):

        # analysum, analysum2 using FFTs
        brad, bphi, bz = B_field        
        bexni = -self.wint * (normal["R_n"] * brad + normal["phi_n"] * bphi + normal["Z_n"] * bz) * 4.0*np.pi*np.pi
        T_p = (T_p_l*bexni).reshape(-1, self.ntheta_stellsym, self.nzeta)
        T_m = (T_m_l*bexni).reshape(-1, self.ntheta_stellsym, self.nzeta)

        T_p = np.pad(T_p, ((0,0), (0,self.ntheta-self.ntheta_stellsym), (0,0)))
        ft_T_p = np.fft.ifft(T_p, axis=1)*self.ntheta
        ft_T_p = np.fft.fft(ft_T_p, axis=2)

        T_m = np.pad(T_m, ((0,0), (0,self.ntheta-self.ntheta_stellsym), (0,0)))        
        ft_T_m = np.fft.ifft(T_m, axis=1)*self.ntheta
        ft_T_m = np.fft.fft(ft_T_m, axis=2)

        kt, kz = np.meshgrid(np.arange(self.ntheta_stellsym), np.arange(self.nzeta))
        i = self.nzeta*kt + kz
        
        num_four = self.mf+self.nf+1
        S_p_4d = np.zeros([num_four, self.ntheta_stellsym, self.nzeta, self.ntheta_stellsym*self.nzeta])
        S_m_4d = np.zeros([num_four, self.ntheta_stellsym, self.nzeta, self.ntheta_stellsym*self.nzeta])

        S_p_4d = put(S_p_4d, Index[:,kt, kz, i], S_p_l.reshape(num_four, self.ntheta_stellsym, self.nzeta)[:,kt,kz])
        S_m_4d = put(S_m_4d, Index[:,kt, kz, i], S_m_l.reshape(num_four, self.ntheta_stellsym, self.nzeta)[:,kt,kz])
        
        # TODO: figure out a faster way to do this, its very sparse
        S_p_4d = np.pad(S_p_4d, ((0,0),(0,self.ntheta-self.ntheta_stellsym),(0,0),(0,0)))
        ft_S_p = np.fft.ifft(S_p_4d, axis=1)*self.ntheta
        ft_S_p = np.fft.fft(ft_S_p, axis=2)

        S_m_4d = np.pad(S_m_4d, ((0,0),(0,self.ntheta-self.ntheta_stellsym),(0,0),(0,0)))
        ft_S_m = np.fft.ifft(S_m_4d, axis=1)*self.ntheta
        ft_S_m = np.fft.fft(ft_S_m, axis=2)

        m,n = np.meshgrid(np.arange(self.mf+1), np.concatenate([np.arange(self.nf+1), np.arange(-self.nf,0)]), indexing="ij")

        I_mn = np.zeros([self.mf+1, 2*self.nf+1])
        I_mn = np.where(np.logical_or(m==0,n==0), (n>=0)*np.sum(self.cmns[:,m,n] * (ft_T_p[:, m, n].imag + ft_T_m[:, m, n].imag), axis=0), I_mn)
        I_mn = np.where(np.logical_and(m != 0, n>0), np.sum(self.cmns[:,m,n] * ft_T_p[:, m, n].imag, axis=0), I_mn)
        I_mn = np.where(np.logical_and(m != 0, n<0), np.sum(self.cmns[:,m,-n] * ft_T_m[:, m, n].imag, axis=0), I_mn)                

        K_mntz = np.zeros([self.mf+1, 2*self.nf+1, self.ntheta_stellsym*self.nzeta])
        K_mntz = np.where(np.logical_or(m==0,n==0)[:,:,np.newaxis], np.sum(self.cmns[:,m,n, np.newaxis] * (ft_S_p[:, m, n, :].imag + ft_S_m[:, m, n, :].imag), axis=0), K_mntz)
        K_mntz = np.where(np.logical_and(m!=0,n>0)[:,:,np.newaxis], np.sum(self.cmns[:,m,n, np.newaxis] * ft_S_p[:, m, n, :].imag,  axis=0), K_mntz)
        K_mntz = np.where(np.logical_and(m!=0,n<0)[:,:,np.newaxis], np.sum(self.cmns[:,m,-n, np.newaxis] * ft_S_m[:, m, n, :].imag, axis=0), K_mntz)
        K_mntz = K_mntz.reshape(self.mf+1, 2*self.nf+1, self.ntheta_stellsym, self.nzeta)
        return I_mn, K_mntz


    def regularizedFourierTransforms(self, K_mntz, B_field, jacobian, normal, coords):

        brad, bphi, bz = B_field        
        bexni = -self.wint * (normal["R_n"] * brad + normal["phi_n"] * bphi + normal["Z_n"] * bz) * 4.0*np.pi*np.pi

        green  = np.zeros([self.ntheta_stellsym, self.nzeta, self.ntheta, self.nzeta, self.nfp_eff])
        greenp = np.zeros([self.ntheta_stellsym, self.nzeta, self.ntheta, self.nzeta, self.nfp_eff])

        # indices over regular and primed arrays
        kt_ip, kz_ip, kt_i, kz_i = np.meshgrid(np.arange(self.ntheta_stellsym), np.arange(self.nzeta), np.arange(self.ntheta), np.arange(self.nzeta), indexing="ij")
        # linear index over primed grid
        ip = (kt_ip*self.nzeta+kz_ip)


        # field-period invariant vectors
        r_squared = (coords["R"]**2 + coords["Z"]**2).reshape((-1,self.nzeta))
        gsave = r_squared[kt_ip, kz_ip] + r_squared - 2.0*coords["Z_sym"][ip].reshape(kt_ip.shape)*coords["Z"].reshape((-1, self.nzeta))
        drv  = -(coords["R_sym"]*normal["R_n"] + coords["Z_sym"]*normal["Z_n"])      
        dsave = drv[ip] + coords["Z"].reshape((-1, self.nzeta))*normal["Z_n"].reshape((self.ntheta_stellsym, self.nzeta))[kt_ip, kz_ip]

        # copy cartesial coordinates in first field period to full domain
        X_full, Y_full = copy_vector_periods(np.array([coords["X"].reshape((-1,self.nzeta))[kt_ip, kz_ip],
                                                       coords["Y"].reshape((-1,self.nzeta))[kt_ip, kz_ip]]),
                                             self.zeta_fp)

        # cartesian components of surface normal
        X_n = (normal["R_n"][ip][:,:,:,:,np.newaxis]*X_full - normal["phi_n"][ip][:,:,:,:,np.newaxis]*Y_full)/coords["R_sym"][ip][:,:,:,:,np.newaxis]
        Y_n = (normal["R_n"][ip][:,:,:,:,np.newaxis]*Y_full + normal["phi_n"][ip][:,:,:,:,np.newaxis]*X_full)/coords["R_sym"][ip][:,:,:,:,np.newaxis]



        ftemp = (gsave[:,:,:,:,np.newaxis]
                 - 2*X_full*coords["X"].reshape((-1, self.nzeta))[np.newaxis,np.newaxis,:,:,np.newaxis]
                 - 2*Y_full*coords["Y"].reshape((-1, self.nzeta))[np.newaxis,np.newaxis,:,:,np.newaxis])
        ftemp = 1/np.where(ftemp<=0, 1, ftemp)
        htemp = np.sqrt(ftemp)
        gtemp = (  coords["X"].reshape((-1, self.nzeta))[np.newaxis,np.newaxis:,:,np.newaxis]*X_n
                 + coords["Y"].reshape((-1, self.nzeta))[np.newaxis,np.newaxis:,:,np.newaxis]*Y_n
                 + dsave[:,:,:,:,np.newaxis])
        greenp_update = ftemp*htemp*gtemp
        green_update = htemp
        mask = ~((self.zeta_fp == 0) | (self.nzeta == 1)).reshape((1,1,1,1,-1,))                
        greenp = np.where(mask, greenp + greenp_update, greenp)
        green  = np.where(mask, green + green_update, green)
               

        if self.nzeta == 1:
            # Tokamak: nfp_eff toroidal "modules"
            delta_kz = (kz_i - kz_ip)%self.nfp_eff
        else:
            # Stellarator: nv toroidal grid points
            delta_kz = (kz_i - kz_ip)%self.nzeta

        # TODO: why is there an additional offset of ntheta?
        delta_kt = kt_i - kt_ip + self.ntheta
        ga1 = self.tanu[delta_kt]*(jacobian["g_tt"][ip]*self.tanu[delta_kt] + 2*jacobian["g_tz"][ip]*self.tanv[delta_kz]) + jacobian["g_zz"][ip]*self.tanv[delta_kz]*self.tanv[delta_kz]
        ga2 = self.tanu[delta_kt]*(jacobian["a_tt"][ip]*self.tanu[delta_kt] +   jacobian["a_tz"][ip]*self.tanv[delta_kz]) + jacobian["a_zz"][ip]*self.tanv[delta_kz]*self.tanv[delta_kz]

        greenp_sing = - (ga2/ga1*1/np.sqrt(ga1))[:,:,:,:,np.newaxis]
        green_sing = - 1/np.sqrt(ga1)[:,:,:,:,np.newaxis]
        mask = ((kt_ip != kt_i) | (kz_ip != kz_i) | (self.nzeta == 1 and kp > 0))[:,:,:,:,np.newaxis] & ((self.zeta_fp == 0) |  (self.nzeta == 1))
        greenp = np.where(mask, greenp + greenp_update + greenp_sing, greenp)
        green = np.where(mask, green + green_update + green_sing, green)                               

        if self.nzeta == 1:
            # Tokamak: need to do toroidal average / integral:
            # normalize by number of toroidal "modules"
            greenp /= self.nfp_eff
            green  /= self.nfp_eff

        greenp = np.sum(greenp, -1)
        green = np.sum(green, -1)
        gstore = np.sum(bexni.reshape((self.ntheta_stellsym, self.nzeta,1,1)) * green[:self.ntheta_stellsym, :, :, :], axis=(0,1))

        # Here, grpmn should contain already the contributions from S^{\pm}_l as computed in analyt/analysum(2).
        # Thus Fourier-transform greenp and add to grpmn.

        # step 1: "fold over" contribution from (pi ... 2pi) in greenp

        # stellarator-symmetric first half-module is copied directly
        # the other half of the first module is "folded over" according to odd symmetry under the stellarator-symmetry operation
        kt, kz = np.meshgrid(np.arange(self.ntheta_stellsym), np.arange(self.nzeta), indexing="ij")
        # anti-symmetric part from stellarator-symmetric half in second half of first toroidal module
        kernel_4d = greenp[:,:,kt, kz] - greenp[:,:,-kt, -kz]

        # accumulated magic from fourp and (sin/cos)mui
        kernel_4d = kernel_4d * 1/self.nfp * (2*np.pi)/self.ntheta * (2.0*np.pi)/self.nzeta
        kernel_4d = put(kernel_4d, Index[:,:,0,:], 0.5*kernel_4d[:,:,0,:])
        kernel_4d = put(kernel_4d, Index[:,:,-1,:], 0.5*kernel_4d[:,:,-1,:])

        kernel_4d = np.pad(kernel_4d, ((0,0),(0,0), (0,self.ntheta-self.ntheta_stellsym),(0,0)))                    
        ft_kernel = np.fft.ifft(kernel_4d, axis=2)*self.ntheta
        ft_kernel = np.fft.fft(ft_kernel, axis=3)

        ft_kernel = np.concatenate([ft_kernel[:self.ntheta_stellsym, :self.nzeta, :self.mf+1, :self.nf+1].imag,
                                    ft_kernel[:self.ntheta_stellsym, :self.nzeta, :self.mf+1, -self.nf:].imag],
                                   axis=-1).transpose((2,3,0,1))
        
        # now assemble final grpmn by adding K_mntz and ft_kernel_4d
        # whoooo... :-)
        grpmn_4d = K_mntz+ ft_kernel


        # first step: "fold over" upper half of gsource to make use of stellarator symmetry
        # anti-symmetric part from stellarator-symmetric half in second half of first toroidal module
        gsource_sym = gstore[kt, kz] - gstore[-kt, -kz]

        # compute Fourier-transform of gsource
        # when only computing the contribution to bvec from gstore, the reference value can be found in potvac,

        # accumulated magic from fouri and (sin/cos)mui
        gsource_sym = gsource_sym * 1/self.nfp * (2*np.pi)/self.ntheta * (2.0*np.pi)/self.nzeta
        gsource_sym = put(gsource_sym, Index[0,:], 0.5*gsource_sym[0,:])
        gsource_sym = put(gsource_sym, Index[-1,:], 0.5*gsource_sym[-1,:])

        gsource_sym = np.pad(gsource_sym, ( (0,self.ntheta-self.ntheta_stellsym),(0,0)))                            
        ft_gsource = np.fft.ifft(gsource_sym, axis=0)*self.ntheta
        ft_gsource = np.fft.fft(ft_gsource, axis=1)
        ft_gsource = np.concatenate([ft_gsource[:self.mf+1,:self.nf+1].imag, ft_gsource[:self.mf+1,-self.nf:].imag], axis=1)

        return ft_gsource, grpmn_4d

    
    def computeScalarMagneticPotential(self, I_mn, ft_gsource, grpmn_4d):

        # compute Fourier transform of grpmn to arrive at amatrix
        grpmn_4d = grpmn_4d * self.wint.reshape([1,1,self.ntheta_stellsym, self.nzeta])
        grpmn_4d = np.pad(grpmn_4d, ((0,0),(0,0), (0,self.ntheta-self.ntheta_stellsym),(0,0)))            
        grpmn_4d = np.fft.ifft(grpmn_4d, axis=2)*self.ntheta
        grpmn_4d = np.fft.fft(grpmn_4d, axis=3)

        amatrix_4d = np.concatenate([grpmn_4d[:, :, :self.mf+1, :self.nf+1].imag, grpmn_4d[:, :, :self.mf+1, -self.nf:].imag], axis=-1)
        # scale amatrix by (2 pi)^2 (#TODO: why ?)
        amatrix_4d *= (2.0*np.pi)**2
        m, n = np.meshgrid(np.arange(self.mf+1), np.arange(2*self.nf+1), indexing="ij")            
        # zero out (m=0, n<0, m', n') modes for all m', n' (#TODO: why ?)
        amatrix_4d = np.where(np.logical_and(m==0, n>self.nf)[:,:,np.newaxis, np.newaxis], 0, amatrix_4d)
        # add diagnonal terms (#TODO: why 4*pi^3 instead of 1 ?)         
        amatrix_4d = put(amatrix_4d, Index[m, n, m, n], amatrix_4d[m,n,m,n] + 4.0*np.pi**3)

        amatrix = amatrix_4d.reshape([(self.mf+1)*(2*self.nf+1), (self.mf+1)*(2*self.nf+1)])

        # combine with contribution from analytic integral; available here in I_mn
        bvec = ft_gsource + I_mn
        # final fixup from fouri: zero out (m=0, n<0) components (#TODO: why ?)
        bvec = put(bvec, Index[0, self.nf+1:], 0.0).flatten()

        potvac = np.linalg.solve(amatrix, bvec)     
        return potvac


    # compute co- and contravariant magnetic field components
    def analyzeScalarMagneticPotential(self, B_field, potvac, jacobian, coords):

        brad, bphi, bz = B_field
        potvac_2d = potvac.reshape([self.mf+1, 2*self.nf+1])
        m_potvac = np.zeros([self.ntheta, self.nzeta]) # m*potvac --> for poloidal derivative
        n_potvac = np.zeros([self.ntheta, self.nzeta]) # n*potvac --> for toroidal derivative

        m,n = np.meshgrid(np.arange(self.mf+1), np.arange(self.nf+1), indexing="ij")

        m_potvac = put(m_potvac, Index[m, n], m * potvac_2d[m, n])
        n_potvac = put(n_potvac, Index[m, n], n * potvac_2d[m, n])
        m_potvac = put(m_potvac, Index[m, -n], m * potvac_2d[m, -n])
        n_potvac = put(n_potvac, Index[m, -n], -n * potvac_2d[m, -n])

        Bp_theta = np.fft.ifft(m_potvac, axis=0) * self.ntheta
        Bp_theta = (np.fft.fft(Bp_theta, axis=1).real[:self.ntheta_stellsym, :]).flatten()

        Bp_zeta = np.fft.ifft(n_potvac, axis=0)*self.ntheta
        Bp_zeta = -(np.fft.fft(Bp_zeta, axis=1).real[:self.ntheta_stellsym, :] * self.nfp).flatten()

        # compute covariant magnetic field components: B_u, B_v
        Bex_theta = coords["R_t"] * brad  + coords["Z_t"] * bz
        Bex_zeta = coords["R_z"] * brad + coords["R_sym"] * bphi + coords["Z_z"] * bz

        vac_field = {}
        vac_field["B_theta"] = Bp_theta + Bex_theta
        vac_field["B_zeta"] = Bp_zeta + Bex_zeta

        # compute B^t, B^z and (with B_t, B_z) then also |B|^2/2

        # TODO: for now, simply copied over from NESTOR code; have to understand what is actually done here!
        h_tz = self.nfp*jacobian["g_tz"]
        h_zz = jacobian["g_zz"]*self.nfp*self.nfp
        det = 1.0/(jacobian["g_tt"]*h_zz-h_tz**2)

        # contravariant components of magnetic field: B^u, B^v

        # B^u
        vac_field["B^theta"] = (h_zz*vac_field["B_theta"] - h_tz*vac_field["B_zeta"])*det

        # B^v
        vac_field["B^zeta"] = (-h_tz * vac_field["B_theta"] + jacobian["g_tt"] * vac_field["B_zeta"])*det

        # |B|^2/2 = (B^u*B_u + B^v*B_v)/2
        vac_field["|B|^2"] = (vac_field["B_theta"] * vac_field["B^theta"] + vac_field["B_zeta"] * vac_field["B^zeta"])/2.0

        # compute cylindrical components B^R, B^\phi, B^Z
        vac_field["BR"]   = coords["R_t"] * vac_field["B^theta"] + coords["R_z"] * vac_field["B^zeta"]
        vac_field["Bphi"] = coords["R_sym"] * vac_field["B^zeta"]
        vac_field["BZ"]   = coords["Z_t"] * vac_field["B^theta"] + coords["Z_z"] * vac_field["B^zeta"]
        return vac_field
        
    def firstIterationPrintout(self, vac_field):
        print("  In VACUUM, np = %2d mf = %2d nf = %2d nu = %2d nv = %2d"%(self.nfp, self.mf, self.nf, self.ntheta, self.nzeta))

        # -plasma current/pi2
        bsubuvac = np.sum(vac_field["B_theta"] * self.wint)*self.signgs*2.0*np.pi
        bsubvvac = np.sum(vac_field["B_zeta"] * self.wint)

        # currents in MA
        fac = 1.0e-6/mu0

        print(("  2*pi * a * -BPOL(vac) = %10.8e \n TOROIDAL CURRENT = %10.8e\n"
              +"  R * BTOR(vac) = %10.8e \n R * BTOR(plasma) = %10.8e")%(bsubuvac*fac, self.ctor*fac, bsubvvac, self.rbtor))

        if self.rbtor*bsubvvac < 0:
            raise ValueError("poloidal current and toroidal field must have same sign, Psi may be incorrect")

        if np.abs((self.ctor - bsubuvac)/self.rbtor) > 1.0e-2:
            raise ValueError("Toroidal current and poloidal field mismatch, boundary may enclose external coil")

    def produceOutputFile(self, vacoutFilename, potvac, vac_field):
        # mode numbers for potvac
        self.xmpot = np.zeros([(self.mf+1)*(2*self.nf+1)])
        self.xnpot = np.zeros([(self.mf+1)*(2*self.nf+1)])
        mn = 0
        for n in range(-self.nf, self.nf+1):
            for m in range(self.mf+1):
                self.xmpot[mn] = m
                self.xnpot[mn] = n*self.nfp
                mn += 1

        vacout = Dataset(vacoutFilename, "w")

        dim_nuv2 = def_ncdim(vacout, self.ntheta_stellsym*self.nzeta)
        dim_mnpd2 = def_ncdim(vacout, (self.mf+1)*(2*self.nf+1))
        dim_mnpd2_sq = def_ncdim(vacout, (self.mf+1)*(2*self.nf+1)*(self.mf+1)*(2*self.nf+1))

        var_bsqvac   = vacout.createVariable("bsqvac", "f8", (dim_nuv2,))
        var_mnpd     = vacout.createVariable("mnpd", "i4")
        var_mnpd2    = vacout.createVariable("mnpd2", "i4")
        var_xmpot    = vacout.createVariable("xmpot", "f8", (dim_mnpd2,))
        var_xnpot    = vacout.createVariable("xnpot", "f8", (dim_mnpd2,))
        var_potvac   = vacout.createVariable("potvac", "f8", (dim_mnpd2,))
        var_brv      = vacout.createVariable("brv", "f8", (dim_nuv2,))
        var_bphiv    = vacout.createVariable("bphiv", "f8", (dim_nuv2,))
        var_bzv      = vacout.createVariable("bzv", "f8", (dim_nuv2,))

        var_bsqvac[:] = vac_field["|B|^2"]
        var_mnpd.assignValue((self.mf+1)*(2*self.nf+1))
        var_mnpd2.assignValue((self.mf+1)*(2*self.nf+1))
        var_xmpot[:] = self.xmpot
        var_xnpot[:] = self.xnpot
        var_potvac[:] = np.fft.fftshift(potvac.reshape([self.mf+1, 2*self.nf+1]), axes=1).T.flatten()
        var_brv[:] = vac_field["BR"]
        var_bphiv[:] = vac_field["Bphi"]
        var_bzv[:] = vac_field["BZ"]

        vacout.close()


def main(vacin_filename, vacout_filename=None, mgrid=None):
    nestor = Nestor(vacin_filename, mgrid)

    # in principle, this needs to be done only once
    nestor.precompute()
    mnmax           = int(nestor.vacin['mnmax'][()])
    xm              = nestor.vacin['xm'][()]
    xn              = nestor.vacin['xn'][()]
    rmnc            = nestor.vacin['rmnc'][()]
    zmns            = nestor.vacin['zmns'][()]
    nzeta           = int(nestor.vacin['nzeta'][()])
    ntheta          = int(nestor.vacin['ntheta'][()])
    ntheta_sym      = ntheta//2 + 1
    nfp             = int(nestor.vacin['nfp'][()])
    # the following calls need to be done on every iteration
    coords = evalSurfaceGeometry_vmec(xm, xn, mnmax, ntheta, nzeta, ntheta_sym, nfp, rmnc, zmns, sym=True)
    normal = compute_normal(coords, nestor.signgs)
    jacobian = compute_jacobian(coords, normal, nestor.nfp)
    B_extern = nestor.interpolateMGridFile(coords["R_sym"], coords["Z_sym"], coords["phi_sym"])

    phiaxis = np.linspace(0,2*np.pi,nestor.nzeta, endpoint=False)/nestor.nfp
    B_plasma = modelNetToroidalCurrent(nestor.raxis_nestor,
                                   phiaxis,
                                   nestor.zaxis_nestor,
                                   nestor.ctor/mu0,
                                   coords["R_sym"],
                                   coords["phi_sym"],
                                       coords["Z_sym"],
                                       nestor.zeta_fp)
    B_field = B_extern + B_plasma
    T_p_l, T_m_l, S_p_l, S_m_l = nestor.compute_T_S(jacobian)
    I_mn, K_mn = nestor.analyticalIntegrals(jacobian, normal, T_p_l, T_m_l, S_p_l, S_m_l, B_field)
    ft_gsource, grpmn_4d = nestor.regularizedFourierTransforms(K_mn, B_field, jacobian, normal, coords)    
    potvac = nestor.computeScalarMagneticPotential(I_mn, ft_gsource, grpmn_4d)
    vac_field = nestor.analyzeScalarMagneticPotential(B_field,
                                          potvac,
                                          jacobian,
                                          coords)
    nestor.firstIterationPrintout(vac_field)

    if vacout_filename is None:
        vacout_filename = vacin_filename.replace("vacin_", "vacout_")
    nestor.produceOutputFile(vacout_filename, potvac, vac_field)
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        vacin_filename = sys.argv[1]
        folder = os.getcwd()
        main(vacin_filename)
    else:
        print("usage: NESTOR.py vacin.nc")

