import math
import numpy as np
from .base import DynamicSystemBase, spaces


class Pendulum(DynamicSystemBase):
    g = 9.8

    def __init__(
        self,
        mass: float = 1.0,
        length: float = 1.0,
        seed: int = None,
        dtype = np.float_,
    ):
        super().__init__(seed=seed,dtype=dtype)
        self._m = mass
        self._L = length
        self._I = self._m * self._L**2 / 3
        self._I_inv = 1 / self._I
        self._H_g = self._m * self.g * self._L / 2
        self._c_g = 3 / 2 * self.g / self._L

        omega_high = 10*math.pi
        U_high = np.array([self._H_g], float)
        X_low = np.array([-1, -1, -omega_high], float)
        X_high = np.array([1, 1, omega_high], float)

        self._A = np.asarray(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, self._c_g, 0],
            ]
        )
        self._B = np.asarray([[0], [0], [self._I_inv]])

        self._X_space = spaces.Box(X_low, X_high,dtype=dtype, seed=seed)
        self._U_space = spaces.Box(-U_high, U_high,dtype=dtype, seed=seed)

        theta0_high = math.radians(30)
        omega0_high = 0.1 * math.pi
        X0_low = np.array([-1, -math.sin(theta0_high), -omega0_high], float)
        X0_high = np.array(
            [-math.cos(theta0_high), math.sin(theta0_high), omega0_high], float
        )
        self._X0_space = spaces.Box(X0_low, X0_high,dtype=dtype, seed=seed)
        # self._X0_space = self._X_space

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray):
        return self._A @ x + self._B @ u
        cosx, sinx, omega = x
        dydt = [omega, self._c_g * math.sin(cosx) + self._I_inv * u[1]]
        return dydt
