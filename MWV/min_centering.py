"""Minimal iterative forward-model coil centering. Coordinate descent in
(x, y, z, tilt_x_deg, tilt_y_deg) minimizing ||B_model(pose) - B_meas||^2."""
import numpy as np


def B(coil, probes, mu0=4e-7 * np.pi):
    """Biot-Savart B at `probes` from a closed polygon `coil` (N+1, 3), 1 A."""
    s = coil[1:] - coil[:-1]
    m = 0.5 * (coil[1:] + coil[:-1])
    R = probes[:, None, :] - m[None, :, :]
    r3 = (R ** 2).sum(-1, keepdims=True) ** 1.5 + 1e-30
    return mu0 / (4 * np.pi) * (np.cross(s, R) / r3).sum(axis=1)


def place(coil, p):
    """Shift+tilt the coil to pose p = (x, y, z, tilt_x_deg, tilt_y_deg).
    Rotation is applied about the coil's own centroid so a pose equal to that
    centroid (with zero tilt) reproduces the input coil unchanged."""
    tx, ty = np.deg2rad(p[3]), np.deg2rad(p[4])
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    c0 = coil.mean(axis=0)
    return (coil - c0) @ (Ry @ Rx).T + p[:3]


def load_dat(path, stride=1):
    """Read a coil .dat file: first line `ncoil s nsec current`, then nsec
    lines of `x y z`. Returns the (N, 3) polygon, optionally subsampled."""
    with open(path) as f:
        f.readline()
        xyz = np.array([line.split() for line in f if line.strip()], dtype=float)
    return xyz[::stride]


def fit(coil, probes, B_meas, init=(0, 0, 0, 0, 0),
        step=(0.05, 0.05, 0.05, 0.5, 0.5), tol=1e-7, max_iter=200):
    """5D coordinate descent with halving step. Returns best pose."""
    p, s = np.array(init, float), np.array(step, float)
    best = ((B(place(coil, p), probes) - B_meas) ** 2).sum()
    for _ in range(max_iter):
        if np.all(s[:3] < tol):
            break
        improved = False
        for i in range(5):
            for d in (+1.0, -1.0):
                t = p.copy(); t[i] += d * s[i]
                v = ((B(place(coil, t), probes) - B_meas) ** 2).sum()
                if v < best:
                    p, best, improved = t, v, True
        if not improved:
            s *= 0.5
    return p


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    coil = load_dat(os.path.join(here, "..", "examples", "Br_3D_example",
                                 "sparc_pf1u.dat"), stride=20)
    c0 = coil.mean(axis=0)
    print(f"coil: {coil.shape[0]} segments, centroid = {c0}")

    # Known perturbation about the coil centroid: 5 mm shift in x, -2 mm in y,
    # 1 mm in z, plus a 0.3° / -0.1° tilt.
    true_pose = np.concatenate([c0 + np.array([0.005, -0.002, 0.001]),
                                [0.30, -0.10]])

    # Dual-shell Hall probe grid sized to the coil
    Rext = np.hypot(coil[:, 0] - c0[0], coil[:, 1] - c0[1])
    R_inner, R_outer = 0.40 * Rext.min(), 0.50 * Rext.max()
    Z_inner = np.linspace(c0[2] - 0.6, c0[2] + 0.6, 14)
    Z_outer = np.linspace(c0[2] - 0.4, c0[2] + 0.4, 8)
    phi_i = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    phi_o = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    PpI, ZzI = np.meshgrid(phi_i, Z_inner, indexing="ij")
    PpO, ZzO = np.meshgrid(phi_o, Z_outer, indexing="ij")
    probes = np.vstack([
        np.column_stack([R_inner * np.cos(PpI.ravel()) + c0[0],
                         R_inner * np.sin(PpI.ravel()) + c0[1], ZzI.ravel()]),
        np.column_stack([R_outer * np.cos(PpO.ravel()) + c0[0],
                         R_outer * np.sin(PpO.ravel()) + c0[1], ZzO.ravel()]),
    ])
    print(f"probes: {probes.shape[0]} ({len(Z_inner)}*{len(phi_i)} inner + "
          f"{len(Z_outer)}*{len(phi_o)} outer)")

    B_meas = B(place(coil, true_pose), probes)
    init = np.concatenate([c0, [0.0, 0.0]])
    rec = fit(coil, probes, B_meas, init=init)
    print(f"true: {true_pose}\nrecv: {rec}\nerr : {rec - true_pose}")
