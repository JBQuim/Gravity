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