import numpy as np
import matplotlib.pyplot as plt
import gravity_tools as grav
from matplotlib.patches import Circle

# param of reference
L = 80  # number of points along the radius of the disc
M = 60  # number of points along the size of square of calculated points
zmax_ref = 4  # number of circle radii to sample over

# param of calculation
N = 100  # number of points along the size of the square of calculated points
zmax = 4  # size of square to sample over
K = 300  # number of subdivisions of the torus's radius
R1 = 2
R2 = 1
mass = 1

# geometry for torus
offsets = np.linspace(-R2, R2, K)
radii_pos = R1 + np.sqrt(R2 ** 2 - offsets ** 2)
radii_neg = R1 - np.sqrt(R2 ** 2 - offsets ** 2)
masses_pos = radii_pos ** 2
masses_neg = -radii_neg ** 2
discs_pos = np.stack((radii_pos, masses_pos, offsets), axis=1)
discs_neg = np.stack((radii_neg, masses_neg, offsets), axis=1)
discs = np.concatenate((discs_pos, discs_neg))
discs[:, 1] = discs[:, 1] / discs[:, 1].sum() * mass
force, potential = grav.multi_disc_grav(L, -zmax_ref, zmax_ref, M, -zmax, zmax, N, discs)

# simple plotting
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(potential, cmap='gray', origin='lower', extent=[-zmax, zmax, -zmax, zmax])
ax.set_aspect('equal')
ax.set_title("Potential")
ax.axis('off')
circ1 = Circle((R1, 0), R2, facecolor='None', edgecolor='k', lw=1, zorder=10)
circ2 = Circle((-R1, 0), R2, facecolor='None', edgecolor='k', lw=1, zorder=10)
ax.add_patch(circ1)
ax.add_patch(circ2)

c = np.linspace(-zmax, zmax, N)
x, y = np.meshgrid(c[5::5], c[5::5])
ax = fig.add_subplot(1, 2, 2)
ax.quiver(x, y, force[5::5, 5::5, 0], force[5::5, 5::5, 1])
ax.set_aspect('equal')
ax.set_title("Gradient")
circ1 = Circle((R1, 0), R2, facecolor='None', edgecolor='k', lw=1, zorder=10)
circ2 = Circle((-R1, 0), R2, facecolor='None', edgecolor='k', lw=1, zorder=10)
ax.add_patch(circ1)
ax.add_patch(circ2)
ax.axis('off')

plt.tight_layout()
plt.savefig('torus.png')
plt.show()