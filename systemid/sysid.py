from collections import OrderedDict
import numpy as np
from numpy import linalg as LA
import abc

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from mathext import *

__DIR_WS = Path(__file__).resolve().parent.parent
from data.dynamics import DOF6Plane, DynamicSystemBase


class Missile(DOF6Plane):
    def __init__(self, seed=None, dtype=np.float_):
        super().__init__(seed=seed, dtype=dtype)

        self._K_png = 1.0  #

    # def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    #     pass

    def step(
        self, u: np.ndarray, dt: float = 0.001, Fs: int = None, Ffix: int = None
    ) -> seeding.Tuple[np.ndarray | bool]:
        return super().step(u, dt, Fs, Ffix)

    def demo():
        print
        pass


if __name__ == "__main__":
    Missile.demo()
