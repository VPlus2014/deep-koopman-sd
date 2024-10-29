import abc
from typing import Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.integrate import odeint, solve_ivp


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
        """(微分方程)状态方程"""
        raise NotImplementedError

    def __init__(
        self,
        seed: int = None,
        dtype=np.float_,  # 积分步长
    ):
        self._seed = seed
        self._np_random, seed = seeding.np_random(self._seed)
        self.dtype = dtype

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

    def _get_obs(self, x: np.ndarray) -> np.ndarray:
        """获取观测值"""
        return x

    def _get_state(self) -> np.ndarray:
        """获取状态"""
        return self._x.copy()

    def _is_terminal(self, x: np.ndarray) -> bool:
        """判断是否满足终止条件"""
        return False

    def reset(self, *, seed: int = None, x0: np.ndarray = None) -> np.ndarray:
        """加载/随机产生合法初始状态, 并返回观测值"""
        if seed is not None:  # 重新初始化随机数种子(用于复刻初始状态)
            self._seed = seed
            self._np_random, seed = seeding.np_random(self._seed)
            self.X0_space.seed(seed)
            self.X_space.seed(seed)
            self.U_space.seed(seed)

        ntry = 0
        if x0 is None:
            while True:
                x0 = self.X0_space.sample()
                ntry += 1
                if ntry == 1000:
                    print(
                        f"Warning: {type(self).__name__}.reset() reset too many times, try again"
                    )
                if not self.X0_space.contains(x0):
                    x0 = None
                    continue
                x0 = self._purify_X_on_reset(x0)
                if self._is_terminal(x0):
                    continue
                break
        assert self.X0_space.contains(x0), f"Invalid initial state {x0}"
        assert not self._is_terminal(x0), f"Initial state {x0} is terminal"
        self._t = 0.0
        self._x = x0
        y = self._get_obs(x0).copy()
        return y

    def _purify_X_on_reset(self, X: np.ndarray) -> np.ndarray:
        """(用于初始化)状态修正合法化"""
        return X

    def _purify_X_on_step(self, X: np.ndarray) -> np.ndarray:
        """(用于ODE修正)状态修正"""
        return X

    def step(
        self,
        u: np.ndarray,
        dt: float = 1e-3,  # 单步积分步长
        Fs: int = None,  # 采样间隔 default=1, 每次观测步进 Fs 步， 共 Fs*dt_integ 物理时长
        Ffix: int = None,  # 修正采样间隔次数 default=Fs, 每积分 Ffix 步， 修正一次状态
    ) -> Tuple[np.ndarray, bool]:
        """（离散时间）状态转移"""
        terminated = False
        Fs = Fs or 1
        Ffix = Ffix or Fs

        t1 = self._t
        x1 = self._x

        df = lambda t, x: self.dynamics(t, x, u)
        x1_ = x1
        for k in range(0, Fs, Ffix):
            t1_ = t1 + k * dt
            ts = [t1_, t1 + (k + 1) * dt]
            sol = solve_ivp(df, ts, x1_, max_step=10 * dt)
            xs = sol.y  # (dimX, Fs+1)

            t2 = ts[-1]
            x2 = xs[:, -1]
            x2 = self._purify_X_on_step(x2)
            x1_ = x2

        self._t = t2
        self._x = x2
        y2 = self._get_obs(x2).copy()
        if self._is_terminal(x2):
            terminated = True
        return y2, terminated

    @classmethod
    def demo(cls):
        print(
            f"Warning: {cls if isinstance(cls, type) else cls.__class__.__name__}.{cls.demo.__name__} Not implemented"
        )
