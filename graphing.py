import numpy as np
import matplotlib.pyplot as plt
import gravity_tools as grav


# param
N = 50
M = 25
R = 1
zmax = 1.5


c = np.linspace(-zmax, zmax, M)
p = np.array(np.meshgrid(c, 0, c)).T.reshape(M, M, 3)
circle, mass = grav.circle_grid(N, R, 1)

force, potential = grav.calc_grav(-zmax, zmax, M, circle, mass)

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.scatter(*circle.T)
ax.scatter(*p.T)
ax.set_title('Masses and samples')
ax.set_xlim(-zmax, zmax)
ax.set_ylim(-zmax, zmax)
ax.set_zlim(-zmax, zmax)

ax = fig.add_subplot(1, 3, 2)
plt.imshow(potential, cmap='inferno')
ax.set_title("Potential")
plt.colorbar()

ax = fig.add_subplot(1, 3, 3)
ax.quiver(force[:, :, 0], force[:, :, 1])
ax.set_aspect('equal')
ax.set_title("Gradient")


plt.tight_layout()
plt.show()