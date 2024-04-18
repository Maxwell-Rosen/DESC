from desc.io import load
from desc.grid import LinearGrid
import numpy as np

eq = load("/Users/xchu/Documents/Courses/G3F/SoftwareEngineering/SLAM/DESC/notebooks/st_pi/mirror_test.h5")

data = eq.compute(
    "<|grad(|B|^2)|/2mu0>_vol",
    grid=LinearGrid(
        rho=np.linspace(0, 1, 30),
        theta=np.linspace(0, 2 * np.pi, 30),
        zeta=np.linspace(0, np.pi, 6),
    ),)