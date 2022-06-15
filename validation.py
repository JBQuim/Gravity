import numpy as np
import matplotlib.pyplot as plt
import gravity_tools as grav


def multi_disc_grav(ref_min, ref_max, ref_res, min, max, res, disks):
    # construct reference
    circle, mass = grav.circle_grid(L, 1, 1)
    force_ref, potential_ref = grav.calc_grav(ref_min, ref_max, ref_res, circle, mass)

    # get input points
    axis = np.linspace(min, max, res)
    points = np.array(np.meshgrid(axis, axis)).T.reshape(res, res, 2)

    # transform input points to map onto reference
    ref_spacing = (ref_max - ref_min) / (ref_res - 1)
    dirs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    offsets = np.stack((disks[:, 2], np.zeros(len(disks))), axis=1)
    points_rel = (points[:, :, np.newaxis] + offsets) / disks[:, 0, np.newaxis]

    # get indices to reference grid
    indices_rel = (points_rel - ref_min) / ref_spacing
    int_indices = indices_rel[..., np.newaxis, :].astype(int) + dirs

    # sanitize indices that are out of range of the reference grid
    mask = np.logical_or(int_indices < 0, int_indices >= ref_res).any(axis=-1)
    points_interp = int_indices * ref_spacing + ref_min  # coords of points used for interpolation
    displacements = int_indices - indices_rel[..., np.newaxis, :]  # distance between interpolation grid and points
    int_indices[mask] = 0  # these are out of range so sanitized

    # look up reference values
    force_disks = force_ref[int_indices[..., 0], int_indices[..., 1]]
    potential_disks = potential_ref[int_indices[..., 0], int_indices[..., 1]]

    # find values not in the reference grid (treat ring as point particle)
    vec = points_interp[mask]
    force_disks[mask] = -vec[..., ::-1] / np.power(np.linalg.norm(vec, axis=-1), 3)[..., np.newaxis]
    potential_disks[mask] = -1 / np.linalg.norm(vec, axis=-1)

    # bilinear interpolation
    areas = (displacements[..., 0] * displacements[..., 1] * np.array([1, -1, -1, 1]))[..., ::-1]
    force_disks = (areas[..., np.newaxis] * force_disks).sum(axis=-2)
    potential_disks = (areas * potential_disks).sum(axis=-1)

    # sum over disks
    force = (force_disks * disks[:, 1, np.newaxis] / disks[:, 0, np.newaxis] ** 2).sum(axis=-2)
    potential = (potential_disks * disks[:, 1] / disks[:, 0] ** 2).sum(axis=-1)
    return force, potential


# param of reference
L = 50  # number of points along the radius of the disc
M = 76  # number of points along the size of square of calculated points
zmax_ref = 5  # number of circle radii to sample over

# param of calculation
N = 60  # number of points along the size of the square of calculated points
zmax = 4  # size of square to sample over
K = 200
R = 2
mass = 1
h = 12
discs = np.stack((np.ones(K) * R, np.ones(K) * mass / K, np.linspace(-h, h, K)), axis=1)

force, potential = multi_disc_grav(-zmax_ref, zmax_ref, M, -zmax, zmax, N, discs)

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
plt.imshow(potential, cmap='inferno', origin='lower')
ax.set_title("Potential")
plt.colorbar()

ax = fig.add_subplot(2, 2, 3)
ax.quiver(force[:, :, 0], force[:, :, 1])
ax.set_aspect('equal')
ax.set_title("Gradient")


c = np.linspace(-zmax, zmax, N)
E = np.zeros_like(c)
inside = np.abs(c) < R
outside = np.abs(c) >= R
density = mass / (2 * h)
E[outside] = 2 * density / c[outside]
E[inside] = 2 * density * c[inside] / (R ** 2)

ax = fig.add_subplot(2, 2, 2)
ax.scatter(c, potential[len(force) // 2 + 1, :], color='k', s=1)
ax.set_title('Potential')

ax = fig.add_subplot(2, 2, 4)
ax.scatter(c, force[len(force) // 2 + 1, :, 0], color='k', s=1)
ax.plot(c, -E)
ax.set_title('Force')

plt.tight_layout()
plt.show()