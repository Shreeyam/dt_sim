"""3D geometry primitives: distances, rotations, frame builders.

Vector helpers and rotation-matrix builders used by the simulator.
"""
import numpy as np


def dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean distance: vector-vector, batch-vector, or batch-batch."""
    if x.ndim == 1 and y.ndim == 1:
        return np.linalg.norm(x - y)
    return np.linalg.norm(x - y, axis=1)


def dist2plane(r: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Signed distance from point(s) p to plane through r with normal n."""
    return (p - r).dot(n) / np.linalg.norm(n)


def rotmat_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def eul2R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation from intrinsic roll/pitch/yaw (rad).

    Composition: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return R_z @ R_y @ R_x


def rotmat_from_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix whose columns are (a_hat, b_hat, a_hat × b_hat).

    Maps the standard basis vectors x,y to a,b. Inputs must be 3D and
    orthogonal; the third column is built from the cross product.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Inputs must be 3-vectors.")

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        raise ValueError("Inputs must be non-zero.")
    a_hat = a / a_norm
    b_hat = b / b_norm

    if not np.isclose(np.dot(a_hat, b_hat), 0.0, atol=1e-8):
        raise ValueError("Inputs must be orthogonal.")
    c_hat = np.cross(a_hat, b_hat)
    return np.column_stack((a_hat, b_hat, c_hat))
