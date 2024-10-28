from collections import OrderedDict
import numpy as np
from numpy import linalg as LA
import abc

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from mathext import *


class DynamicSystemBase(abc.ABC):
    """Abstract base class for dynamic systems."""

    _X0_space: spaces.Box
    """初始状态空间"""
    _X_space: spaces.Box
    """状态空间"""
    _U_space: spaces.Box = None
    """控制输入空间"""

    @abc.abstractmethod
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """状态方程"""
        raise NotImplementedError

    def __init__(self, seed: int = None, dtype=np.float_):
        self._seed = seed
        self.dtype = dtype
        self._rng, seed = seeding.np_random(seed)

    @property
    def X0_space(self) -> spaces.Box:
        """初始状态空间"""
        try:
            return self._X0_space
        except AttributeError as e:
            raise NotImplementedError(f"{type(self).__name__}._X0_space") from e

    @property
    def X_space(self) -> spaces.Box:
        """状态空间"""
        try:
            return self._X_space
        except AttributeError as e:
            raise NotImplementedError(f"{type(self).__name__}._X_space") from e

    @property
    def U_space(self) -> spaces.Box:
        """控制输入空间"""
        return self._U_space

    @classmethod
    def demo(cls):
        print(
            f"Warning: {cls if isinstance(cls, type) else cls.__class__.__name__}.{cls.demo.__name__} Not implemented"
        )


class DOF6PlaneQuat(DynamicSystemBase):
    """NED dof6 quaternion plane dynamics."""

    g = 9.81  # gravity constant

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
        Moment_max: float = 1000.0,  # N*m
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
            ]
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

        X_max = np.inf
        X_named = OrderedDict(
            [
                ("X_e", spaces.Box(-X_max, X_max, dtype=dtype, seed=seed)),
                ("Y_e", spaces.Box(-X_max, X_max, dtype=dtype, seed=seed)),
                ("Z_e", spaces.Box(-X_max, X_max, dtype=dtype, seed=seed)),
                ("U", spaces.Box(-V_max, V_max, dtype=dtype, seed=seed)),
                ("V", spaces.Box(-V_max, V_max, dtype=dtype, seed=seed)),
                ("W", spaces.Box(-V_max, V_max, dtype=dtype, seed=seed)),
                ("P", spaces.Box(-omega_max, omega_max, dtype=dtype, seed=seed)),
                ("Q", spaces.Box(-omega_max, omega_max, dtype=dtype, seed=seed)),
                ("R", spaces.Box(-omega_max, omega_max, dtype=dtype, seed=seed)),
                ("q1", spaces.Box(-1, 1, dtype=dtype, seed=seed)),
                ("q2", spaces.Box(-1, 1, dtype=dtype, seed=seed)),
                ("q3", spaces.Box(-1, 1, dtype=dtype, seed=seed)),
                ("q4", spaces.Box(-1, 1, dtype=dtype, seed=seed)),
            ]
        )

        self._X_space_named = spaces.Dict(X_named)
        self._U_space_named = spaces.Dict(
            OrderedDict(
                [
                    ("nx", spaces.Box(-nx_b_max, nx_f_max, dtype=dtype, seed=seed)),
                    ("M1", spaces.Box(-Moment_max, Moment_max, dtype=dtype, seed=seed)),
                    ("M2", spaces.Box(-Moment_max, Moment_max, dtype=dtype, seed=seed)),
                    ("M3", spaces.Box(-Moment_max, Moment_max, dtype=dtype, seed=seed)),
                ]
            )
        )

        self._X_space = spaces.flatten_space(self._X_space_named)
        self._U_space = spaces.flatten_space(self._U_space_named)

        self._e1 = np.array([1, 0, 0], dtype=dtype)
        self._e3 = np.array([0, 0, 1], dtype=dtype)

        X0_named = X_named.copy()
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

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xyz, uvw, pqr, Q_eb = np.split(x, self._X_split)
        q0, qv = np.split(Q_eb, [1])
        nx, lmn = np.split(u, [1])
        nx = nx.item()
        V = LA.norm(uvw)
        # 气动力
        qbar = 0.5 * self.rho * V**2
        e_v = uvw / max(V, 1e-4)
        D_b = (-qbar * self._C_D) * e_v
        L_b = (-qbar * self._C_L) * self._e3

        dot_xyz = quat_rot(Q_eb, uvw)

        dot_uvw = (
            self._m_inv * (L_b + D_b) + (nx * self.g) * self._e1 - np.cross(pqr, uvw)
        )
        dot_pqr = self._I_inv @ (lmn - np.cross(pqr, self._I_b @ pqr))

        # $\dot{Q}_{eb}=Q_{eb}*(0, 0.5*\omega_{/b})
        u_b = 0.5 * pqr
        dot_q0 = -np.dot(qv, u_b)
        dot_qv = q0 * u_b + np.cross(qv, u_b)

        # print(int(t / 1e-3), LA.norm(Q_be))
        dotX = np.concatenate([dot_xyz, dot_uvw, dot_pqr, [dot_q0], dot_qv])
        assert dotX.shape == x.shape
        return dotX

    @staticmethod
    def speed_is_too_low(x: np.ndarray) -> bool:
        return LA.norm(x[:3]) < 0.1

    @staticmethod
    def demo():
        from scipy.integrate import odeint

        seed = None
        rng = np.random.default_rng(seed)

        m = DOF6PlaneQuat()
        x0 = m.X_space.sample()
        u0 = m.U_space.sample()

        # random quaternion
        n_ = 1 - np.random.rand(3)
        n_ = n_ / LA.norm(n_)
        t = np.random.rand() * 2 * np.pi
        q = [np.cos(t / 2), *(np.sin(t / 2) * n_)]
        x0[-4:] = q

        n_steps = 100
        dt = 1e-3
        us = [u0]
        uk = u0
        dt_sqrt = np.sqrt(dt)
        for k in range(n_steps):
            dw = (0.1 * dt_sqrt) * rng.normal(size=uk.shape)
            du = uk + 0.9 * (0 - uk) * dt + dw
            uk = np.clip(uk + du, m.U_space.low, m.U_space.high)
            us.append(uk)

        dyna = lambda t, x: m.dynamics(t, x, us[int(t / dt)])

        ys: np.ndarray = odeint(
            dyna, x0, np.arange(n_steps) * dt, tfirst=True, hmax=dt
        )  # (L,D)
        assert ys.shape == (n_steps, x0.shape[-1])


class Missile(DOF6PlaneQuat):
    def __init__(self, seed=None, dtype=np.float_):
        super().__init__(seed=seed, dtype=dtype)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass


if __name__ == "__main__":
    DOF6PlaneQuat.demo()
