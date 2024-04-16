# need mayavi
from mayavi import mlab
import numpy as np

mesh_LCFS = np.load("./mesh_LCFS.npy")
mesh_end0 = np.load("./mesh_end0.npy")
mesh_end1 = np.load("./mesh_end1.npy")
lines = np.load("./lines.npy")
surfs = np.load("./surfs.npy")
cons_theta = np.load("./constant_theta.npy")

mlab.figure(size=(1080 * 2, 720 * 2), bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.6, 0.1, 0.1))
# mlab.mesh(mesh_LCFS[0], mesh_LCFS[1], mesh_LCFS[2], color=(.8,.8,.8))
mlab.mesh(
    mesh_LCFS[0] * np.cos(mesh_LCFS[1]),
    mesh_LCFS[0] * np.sin(mesh_LCFS[1]),
    mesh_LCFS[2],
    scalars=mesh_LCFS[3],
)
# mlab.mesh(mesh_end0[0], mesh_end0[1], mesh_end0[2], color=(.8,.8,.8))
mlab.mesh(
    mesh_end0[0] * np.cos(mesh_end0[1]),
    mesh_end0[0] * np.sin(mesh_end0[1]),
    mesh_end0[2],
    scalars=mesh_end0[3],
)
# mlab.mesh(mesh_end1[0], mesh_end1[1], mesh_end1[2], color=(.8,.8,.8))
mlab.mesh(
    mesh_end1[0] * np.cos(mesh_end1[1]),
    mesh_end1[0] * np.sin(mesh_end1[1]),
    mesh_end1[2],
    scalars=mesh_end1[3],
)

for line in lines:
    mlab.plot3d(line[0]*np.cos(line[1]), line[0]*np.sin(line[1]), line[2], tube_radius=None)
for surf in surfs:
    mlab.plot3d(surf[0]*np.cos(surf[1]), surf[0]*np.sin(surf[1]), surf[2], tube_radius=None)
for cons in cons_theta:
    mlab.plot3d(cons[0]*np.cos(cons[1]), cons[0]*np.sin(cons[1]), cons[2], tube_radius=None)

mlab.show()
