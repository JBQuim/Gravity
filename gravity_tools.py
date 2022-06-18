import numpy as np


def find_force(points, mass_pos, masses=1, eps=1e-2):
    vector = points[..., np.newaxis, :] - mass_pos[np.newaxis, :]
    size = np.linalg.norm(vector, axis=-1)
    size[np.abs(size) <= eps] = np.nan
    force = np.sum(- masses[:, np.newaxis] * vector / np.power(size, 3)[..., np.newaxis], axis=-2)
    return force[..., np.array([0, 2])]


def find_potential(points, mass_pos, masses=1, eps=1e-5):
    vector = points[..., np.newaxis, :] - mass_pos[np.newaxis, :]
    size = np.linalg.norm(vector, axis=-1)
    size[size <= eps] = np.nan
    potential = (-masses / size).sum(axis=-1)
    return potential


def calc_grav(min, max, N, mass_points, masses):
    c = np.linspace(min, max, N)
    p = np.array(np.meshgrid(c, 0, c)).T.reshape(N, N, 3)
    return find_force(p, mass_points, masses), find_potential(p, mass_points, masses)


def ring(N, R, mass):
    pos = np.zeros((N, 3))
    masses = np.ones(N) / mass
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos[:, 0], pos[:, 1] = R * np.cos(theta), R * np.sin(theta)
    return pos, masses


def circle_rad(N_ring, N_rad, R, mass):
    pos = np.zeros((N_rad, N_ring, 3))
    r = np.linspace(R, 0, N_rad, endpoint=False)[::-1]
    masses = (np.tile(np.arange(0, N_rad), (N_ring, 1)).T * 2 + 1) * mass / N_rad ** 2 / N_ring
    theta = np.linspace(0, 2 * np.pi, N_ring, endpoint=False)
    pos[..., 0], pos[..., 1] = r[:, np.newaxis] * np.cos(theta), r[:, np.newaxis] * np.sin(theta)
    return pos.reshape(-1, 3), masses.flatten()


def circle_grid(N, R, mass):
    x = np.linspace(-R, R, N)
    grid = np.array(np.meshgrid(x, x, 0)).T.reshape(N, N, 3)
    part = (grid ** 2).sum(axis=-1) <= R ** 2
    pos = grid[part]
    return pos, np.ones(len(pos)) / mass / len(pos)


def multi_disc_grav(ref_L, ref_min, ref_max, ref_res, min, max, res, disks):
    # construct reference
    circle, mass = circle_grid(ref_L, 1, 1)
    force_ref, potential_ref = calc_grav(ref_min, ref_max, ref_res, circle, mass)

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