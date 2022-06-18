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
zmax = 3  # size of square to sample over
K = 500  # number of subdivisions of the sphere's radius
R = 1
mass = 1
offsets = np.linspace(-R, R, K + 1, endpoint=False)[1:]
radii = np.sqrt(R ** 2 - offsets ** 2)
masses = (R ** 2 - offsets ** 2)
masses = masses / masses.sum() * mass
discs = np.stack((radii, masses, offsets), axis=1)
force, potential = grav.multi_disc_grav(L, -zmax_ref, zmax_ref, M, -zmax, zmax, N, discs)

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
plt.imshow(potential, cmap='gray', origin='lower', extent=[-zmax, zmax, -zmax, zmax])
ax.set_aspect('equal')
ax.set_title("Potential")
ax.axis('off')
circ = Circle((0, 0), R, facecolor='None', edgecolor='k', lw=1, zorder=10)
ax.add_patch(circ)

c = np.linspace(-zmax, zmax, N)
x, y = np.meshgrid(c[5::10], c[5::10])
ax = fig.add_subplot(2, 2, 3)
ax.quiver(x, y, force[5::10, 5::10, 0], force[5::10, 5::10, 1])
ax.set_aspect('equal')
ax.set_title("Gradient")
ax.axis('off')
circ = Circle((0, 0), R, facecolor='None', edgecolor='k', lw=1, zorder=10)
ax.add_patch(circ)

ax = fig.add_subplot(2, 2, 2)
ax.plot(c, potential[len(force) // 2, :], color='k')
ax.plot(c, potential[:, len(force) // 2], color='k', linestyle='dotted')
ax.plot(c * np.sqrt(2), potential[np.arange(N), np.arange(N)], color='k', linestyle='dashed')
ax.set_xlim(-zmax, zmax)
ax.set_title('Potential')

ax = fig.add_subplot(2, 2, 4)
ax.plot(c, force[len(force) // 2, :, 0], color='k')
ax.plot(c, force[:, len(force) // 2, 1], color='k', linestyle='dotted')
ax.set_title('Force')

plt.tight_layout()
plt.savefig('sphere.png')
plt.show()
