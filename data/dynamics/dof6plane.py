from collections import OrderedDict
import numpy as np
from numpy import linalg as LA
from .base import DynamicSystemBase, spaces
from .mathext import *

_PI = math.pi
_2PI = _PI
_PI_HF = _PI * 0.5


class DOF6Plane(DynamicSystemBase):
    """NED dof6 quaternion plane dynamics."""

    _g = 9.81  # gravity constant

    def __init__(
        self,
        mass: float = 1000.0,
        I_xx: float = 200.0,
        I_yy: float = 200.0,
        I_zz: float = 200.0,
        I_xy: float = 0.0,
        I_yz: float = 0.0,
        I_xz: float = 0.0,
        V_max: float = 1.5 * 334,  # m/s
        omega_max: float = 2 * np.pi / 4,  # rad/s
        Moment_max: float = 200.0,  # N*m
        nx_f_max: float = 8.0,  # G
        nx_b_max: float = 0.5,  # G
        dtype=np.float_,
        seed=None,
    ):
        super().__init__(seed=seed, dtype=dtype)
        I_b = np.array(
            [
                [I_xx, I_xy, I_xz],
                [I_xy, I_yy, I_yz],
                [I_xz, I_yz, I_zz],
            ],
            dtype=dtype,
        )
        I_inv = np.linalg.inv(I_b)
        self._I_b = I_b
        self._I_inv = I_inv
        self._mass = mass
        self._m_inv = 1.0 / mass
        self._V_max = V_max
        self._omega_max = omega_max
        self._Moment_max = Moment_max
        self._C_L = 1.4  # lift coefficient
        self._C_D = 0.3  # drag coefficient

        x_max = 200e3  # \approx 200km
        x0_max = min(10.0, x_max * 1e-3)
        X_named = OrderedDict(
            [
                # p_{/e}
                ("X_e", spaces.Box(-x_max, x_max, dtype=dtype, seed=seed)),
                ("Y_e", spaces.Box(-x_max, x_max, dtype=dtype, seed=seed)),
                ("Z_e", spaces.Box(-x_max, x_max, dtype=dtype, seed=seed)),
                # v_{/b}
                ("U", spaces.Box(-V_max, V_max, dtype=dtype, seed=seed)),
                ("V", spaces.Box(-V_max, V_max, dtype=dtype, seed=seed)),
                ("W", spaces.Box(-V_max, V_max, dtype=dtype, seed=seed)),
                # \omega_{/b}
                ("P", spaces.Box(-omega_max, omega_max, dtype=dtype, seed=seed)),
                ("Q", spaces.Box(-omega_max, omega_max, dtype=dtype, seed=seed)),
                ("R", spaces.Box(-omega_max, omega_max, dtype=dtype, seed=seed)),
                # Q_{eb}
                ("q0", spaces.Box(-1, 1, dtype=dtype, seed=seed)),
                ("q1", spaces.Box(-1, 1, dtype=dtype, seed=seed)),
                ("q2", spaces.Box(-1, 1, dtype=dtype, seed=seed)),
                ("q3", spaces.Box(-1, 1, dtype=dtype, seed=seed)),
            ]
        )

        self._X_space_named = spaces.Dict(X_named)
        self._U_space_named = spaces.Dict(
            OrderedDict(
                [
                    ("nx", spaces.Box(-nx_b_max, nx_f_max, dtype=dtype, seed=seed)),
                    # M_{/b}
                    ("L", spaces.Box(-Moment_max, Moment_max, dtype=dtype, seed=seed)),
                    ("M", spaces.Box(-Moment_max, Moment_max, dtype=dtype, seed=seed)),
                    ("N", spaces.Box(-Moment_max, Moment_max, dtype=dtype, seed=seed)),
                ]
            )
        )

        self._X_space = spaces.flatten_space(self._X_space_named)
        self._U_space = spaces.flatten_space(self._U_space_named)

        self._e1 = np.array([1, 0, 0], dtype=dtype)
        self._e3 = np.array([0, 0, 1], dtype=dtype)

        X0_named = X_named.copy()
        X0_named["X_e"] = spaces.Box(-x0_max, x0_max, dtype=dtype, seed=seed)
        X0_named["Y_e"] = spaces.Box(-x0_max, x0_max, dtype=dtype, seed=seed)
        X0_named["Z_e"] = spaces.Box(-x0_max, x0_max, dtype=dtype, seed=seed)
        X0_named["U"] = spaces.Box(120, 240, dtype=dtype, seed=seed)
        X0_named["V"] = spaces.Box(0, 0, dtype=dtype, seed=seed)
        X0_named["W"] = spaces.Box(0, 0, dtype=dtype, seed=seed)
        X0_named["P"] = spaces.Box(0, 0, dtype=dtype, seed=seed)
        X0_named["Q"] = spaces.Box(0, 0, dtype=dtype, seed=seed)
        X0_named["R"] = spaces.Box(0, 0, dtype=dtype, seed=seed)
        self._X0_space_named = spaces.Dict(X0_named)
        self._X0_space = spaces.flatten_space(self._X0_space_named)

    @property
    def rho(self):
        return 1.225  # kg/m^3

    _X_dims = [3, 3, 3, 4]
    _X_split = np.cumsum(_X_dims)[:-1]
    _e1 = np.array([1, 0, 0], np.float_)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xyz, uvw, pqr, Q_eb = np.split(x, self._X_split)
        Q_eb = quat_normalize(Q_eb)  # 归一化
        reQ = quat_Re(Q_eb)
        imQ = quat_Im(Q_eb)
        nx, lmn = np.split(u, [1])
        nx = nx.item()

        # 气动力模型
        V = LA.norm(uvw)
        rho_hf_V = 0.5 * self.rho * V
        qbar = rho_hf_V * V

        D_b = (-rho_hf_V * self._C_D) * uvw
        L_b = (-qbar * self._C_L) * self._e3

        g = self._g

        dot_xyz = quat_rot(Q_eb, uvw)
        dot_uvw = self._m_inv * (L_b + D_b) + (nx * g) * self._e1 - np.cross(pqr, uvw)
        dot_pqr = self._I_inv @ (lmn - np.cross(pqr, self._I_b @ pqr))
        w = 0.5 * pqr
        dot_q0 = -np.dot(imQ, w)
        dot_qv = reQ * pqr + np.cross(imQ, w)

        # print(int(t / 1e-3), LA.norm(Q_be))
        dotX = np.concatenate(
            [dot_xyz, dot_uvw, dot_pqr, [dot_q0], dot_qv], dtype=self.dtype
        )
        assert dotX.shape == x.shape
        return dotX

    def _purify_X_on_reset(self, X: np.ndarray) -> np.ndarray:
        # 姿态若表示为四元数/旋转矩阵，则
        #   初始化时会面临在盒子空间上的低维嵌入问题，随机遍历很难出合法值
        # 若表示为欧拉角，则
        #   误差计算需要模 2pi 区间， 且合理性不如 SO3 上的误差
        #   全姿态仿真面临万向节死锁
        # 随机欧拉角->四元数
        r = 1e-1
        w = self._np_random.random(size=3) * r
        rpy_deg = [-180, -90, -180] + (0.5 + w) * [360, 180, 360]
        rpy_deg = rpy_deg.astype(np.int_)
        Q_eb = rpy2quat(*rpy_deg, degrees=True)

        X[-4:] = Q_eb
        return X

    def _purify_X_on_step(self, X: np.ndarray) -> np.ndarray:
        Q_eb = X[-4:]
        #
        # qn = quat_norm(Q_eb)
        # err = np.abs(qn - 1)
        # if err > 1e-2:
        #     print(f"|Q_eb| is up to {err}")
        #
        Q_eb = quat_normalize(Q_eb)
        X[-4:] = Q_eb
        return X

    def _is_terminal(self, X: np.ndarray) -> bool:
        return self.speed_is_too_low(X)

    def _get_obs(self, X: np.ndarray) -> np.ndarray:
        """为了方便计算旋转误差,将四元数转为旋转矩阵, 因为 |U-V|_F=|U^T V-I|_F"""
        X_, Q_eb = X[:-4], X[-4:]
        T_eb = quat2mat(Q_eb)
        y = np.concatenate([X_, T_eb.flatten()])
        return y

    @staticmethod
    def speed_is_too_low(X: np.ndarray) -> bool:
        return LA.norm(X[:3]) < 0.1

    @staticmethod
    def demo(dt=1e-3, Fs=10, seed=None):
        from time import time

        t0 = time()

        m = DOF6Plane(dtype=np.float64, seed=seed)
        y0 = m.reset()
        ys = [y0]
        term = False
        trunc = False
        k = 0
        while not (term or trunc):
            u = m.U_space.sample()
            y1, term = m.step(u, Fs=Fs, dt=dt)
            ys.append(y1)
            k += 1
            trunc = k > 100
        print(f"Terminated after {k} steps, wall time: {time() - t0:.3f}s")


if __name__ == "__main__":
    raise Exception("This file is not meant to be run directly.")
