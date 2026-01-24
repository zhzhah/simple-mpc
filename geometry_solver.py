import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class GeometryResult:
    roll: float
    pitch: float
    toe_offsets: np.ndarray


class GeometryAdaptor:
    def __init__(
        self,
        foot_pos_base,
        axle_dir_base,
        toe_bounds=0.05,
        roll_pitch_bounds=0.5,
    ):
        self.foot_pos_base = np.asarray(foot_pos_base, dtype=float)
        self.axle_dir_base = np.asarray(axle_dir_base, dtype=float)
        self.toe_bounds = float(toe_bounds)
        self.roll_pitch_bounds = float(roll_pitch_bounds)

    def _rot_x(self, r):
        c, s = np.cos(r), np.sin(r)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rot_y(self, p):
        c, s = np.cos(p), np.sin(p)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rot_z(self, y):
        c, s = np.cos(y), np.sin(y)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def solve(self, v_cmd, omega_cmd, yaw=0.0):
        v_cmd = float(v_cmd)
        omega_cmd = float(omega_cmd)
        yaw = float(yaw)

        if abs(omega_cmd) < 1e-6:
            return GeometryResult(0.0, 0.0, np.zeros(4))

        icr_local = np.array([0.0, v_cmd / omega_cmd, 0.0])
        R_yaw = self._rot_z(yaw)

        def cost(x):
            roll, pitch = x[0], x[1]
            toe = x[2:]
            R = R_yaw @ self._rot_y(pitch) @ self._rot_x(roll)
            r_O_star = R @ icr_local
            total = 0.0
            for i in range(4):
                p = self.foot_pos_base[i].copy()
                p[0] += toe[i]
                r_contact = R @ p
                a_y = R @ self.axle_dir_base[i]
                g_z = np.array([0.0, 0.0, 1.0])
                c_x = np.cross(a_y, g_z)
                n = np.linalg.norm(c_x)
                if n < 1e-9:
                    continue
                c_x /= n
                rho = r_O_star - r_contact
                rho_n = np.linalg.norm(rho)
                if rho_n < 1e-9:
                    continue
                e = np.dot(rho, c_x) / rho_n
                total += e * e
            return total

        x0 = np.zeros(6)
        bounds = [
            (-self.roll_pitch_bounds, self.roll_pitch_bounds),
            (-self.roll_pitch_bounds, self.roll_pitch_bounds),
        ] + [(-self.toe_bounds, self.toe_bounds)] * 4

        res = minimize(cost, x0, method="SLSQP", bounds=bounds, options={"maxiter": 50})
        x = res.x if res.success else x0
        return GeometryResult(x[0], x[1], x[2:])
