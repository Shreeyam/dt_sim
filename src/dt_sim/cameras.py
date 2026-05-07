"""Camera intrinsics, projection, and unprojection."""
import datetime

import numpy as np

from .geometry import eul2R, rotmat_from_vec
from .orbits import ecef2eci_vec, horizon_time, kepler2eci, propagate_orbit


def get_intrinsics(f, c_x, c_y) -> np.ndarray:
    K = np.hstack([
        np.array([-f, 0, c_x, 0, f, c_y, 0, 0, 1]).reshape(3, 3),
        np.zeros((3, 1)),
    ])
    return K


def get_intrinsics_from_fov(fov: float, width: int, height: int,
                            axis: str = "x") -> np.ndarray:
    if axis == "x":
        f = width / (2 * np.tan(np.deg2rad(fov / 2)))
    elif axis == "y":
        f = height / (2 * np.tan(np.deg2rad(fov / 2)))
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return get_intrinsics(f, width / 2, height / 2)


def get_extrinsics(R_t: np.ndarray, p: np.ndarray) -> np.ndarray:
    R = R_t @ np.eye(3)
    t = -R @ p
    Rt = np.hstack([R, t.reshape(3, 1)])
    return np.vstack([Rt, np.array([[0, 0, 0, 1]])])


def get_camera_matrix(K: np.ndarray, R: np.ndarray, p: np.ndarray) -> np.ndarray:
    return K @ get_extrinsics(R, p)


def project(P: np.ndarray, points: np.ndarray, z_clip: bool = True) -> np.ndarray:
    pts = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    pts = (P @ pts.T).T
    if z_clip:
        pts = pts[pts[:, 2] > 0]
    pts = pts / pts[:, [2]]
    return pts[:, 0:2]


def _camera_pose_from_orbit(orbit, time, roll_angle: float, pitch_angle: float):
    """Build the camera-to-world rotation and origin for an orbit pose."""
    r, v = kepler2eci(propagate_orbit(orbit, time))
    r_unit = r / np.linalg.norm(r)
    v_unit = v / np.linalg.norm(v)
    R_t = rotmat_from_vec(r_unit, -v_unit)
    R_q = eul2R(0, np.deg2rad(roll_angle), np.deg2rad(pitch_angle))
    R = np.linalg.inv(R_t @ R_q)[[2, 1, 0], :]
    return R, r


def project_from_orbit(points: np.ndarray, K: np.ndarray, orbit, time,
                       roll_angle: float = 0, pitch_angle: float = 15) -> np.ndarray:
    R, r = _camera_pose_from_orbit(orbit, time, roll_angle, pitch_angle)
    P = get_camera_matrix(K, R, r)
    if points.size > 0:
        return project(P, points, False)
    return np.zeros((0, 2))


def unproject(P: np.ndarray, img_points: np.ndarray, depth) -> np.ndarray:
    """Inverse of project, given depth.

    Image points are 2D pixel coords; depth is camera-frame z (scalar or per-point).
    Returns world-frame 3D points. Used for figure/video work, not the simulator.
    """
    if np.isscalar(depth):
        depth = np.full((img_points.shape[0], 1), depth)
    else:
        depth = np.asarray(depth).reshape(-1, 1)
    M = P[:, :3]
    p4 = P[:, 3]
    homogeneous = np.hstack([img_points * depth, depth])
    return (homogeneous - p4) @ np.linalg.inv(M).T


def unproject_from_orbit(img_points: np.ndarray, depth, K: np.ndarray, orbit,
                         time, roll_angle: float = 0,
                         pitch_angle: float = 15) -> np.ndarray:
    R, r = _camera_pose_from_orbit(orbit, time, roll_angle, pitch_angle)
    P = get_camera_matrix(K, R, r)
    return unproject(P, img_points, depth)


def project_in_box(pitch_deg: float, roll_deg: float, orbit, t,
                   accesses, points: np.ndarray, width: int, height: int,
                   K: np.ndarray):
    """Project ECEF points into the camera frame; return those inside the FoV."""
    points_eci = ecef2eci_vec(np.atleast_2d(points), t)
    proj = project_from_orbit(points_eci, K, orbit, t,
                              pitch_angle=pitch_deg, roll_angle=roll_deg)
    in_box_idx = np.array([
        i for i, p in enumerate(proj)
        if 0 <= p[0] <= width and 0 <= p[1] <= height
    ])
    in_box_accesses = [a for i, a in enumerate(accesses) if i in in_box_idx]
    return in_box_accesses, in_box_idx, proj


def filter_accesses_horizon(orbit, time: datetime.datetime, accesses,
                            pos_ecef: np.ndarray, field_of_regard: float = 30):
    """Keep accesses inside the horizon time window and within the FoR cone."""
    horizon = datetime.timedelta(seconds=horizon_time(orbit))
    return [
        (r, a, t, access, idx)
        for r, a, t, access, idx in accesses
        if time <= t <= time + horizon and -field_of_regard <= a <= field_of_regard
    ]


def create_box(width: int, height: int, points_per_edge: int = 0) -> np.ndarray:
    """Pixel-rectangle outline as densified (x, y) points, closed at corner 0.

    Used by figure scripts to draw the camera FoV polygon on the ground.
    """
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    edges = []
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        edge_points = np.linspace(start, end, points_per_edge)
        edges.append(edge_points[:-1])
    return np.vstack(edges + [corners[0]])


def ecef2pitchroll(pos_ecef: np.ndarray, v_ecef: np.ndarray, vec: np.ndarray):
    """Decompose an ECEF target vector into camera (pitch, roll) degrees."""
    Up = pos_ecef / np.linalg.norm(pos_ecef)
    Along = v_ecef / np.linalg.norm(v_ecef)
    Right = np.cross(Along, Up)
    Right = Right / np.linalg.norm(Right)
    Along = np.cross(Up, Right)
    Along = Along / np.linalg.norm(Along)

    R_ecef_to_body = np.vstack([Right, Along, Up])
    v_local = R_ecef_to_body @ vec
    v_local_norm = v_local / np.linalg.norm(v_local)

    pitch = -np.arctan2(v_local_norm[1], v_local_norm[2])
    roll = -np.arctan2(v_local_norm[0], v_local_norm[2])
    return np.degrees(pitch), np.degrees(roll)
