import numpy as np
import matplotlib.pyplot as plt
import gravity_tools as grav

# param of reference
L = 50  # number of points along the radius of the disc
M = 76  # number of points along the size of square of calculated points
zmax_ref = 5  # number of circle radii to sample over

# param of calculation
N = 100  # number of points along the size of the square of calculated points
zmax = 4  # size of square to sample over
K = 200
R = 2
mass = 1
h = 12
discs = np.stack((np.ones(K) * R, np.ones(K) * mass / K, np.linspace(-h, h, K)), axis=1)

force, potential = grav.multi_disc_grav(L, -zmax_ref, zmax_ref, M, -zmax, zmax, N, discs)

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
plt.imshow(potential, cmap='gray', origin='lower', extent=[-zmax, zmax, -zmax, zmax])
ax.set_title("Potential")
ax.axis('off')
ax.plot([-R, -R], [zmax, -zmax], color='k')
ax.plot([R, R], [zmax, -zmax], color='k')

c = np.linspace(-zmax, zmax, N)
x, y = np.meshgrid(c[5::10], c[5::10])
ax = fig.add_subplot(2, 2, 3)
ax.quiver(x, y, force[5::10, 5::10, 0], force[5::10, 5::10, 1])
ax.set_aspect('equal')
ax.set_title("Gradient")
ax.axis('off')
ax.plot([-R, -R], [zmax, -zmax], color='k')
ax.plot([R, R], [zmax, -zmax], color='k')

c = np.linspace(-zmax, zmax, N)

ax = fig.add_subplot(2, 2, 2)
ax.plot(c, potential[len(force) // 2 + 1, :], color='k')
ax.set_title('Potential')

ax = fig.add_subplot(2, 2, 4)
ax.plot(c, force[len(force) // 2 + 1, :, 0], color='k')
ax.set_title('Force')

plt.tight_layout()
plt.savefig('cylinder.png')
plt.show()