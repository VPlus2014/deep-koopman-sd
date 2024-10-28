import abc

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


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

    def __init__(self, seed: int = None,dtype=np.float_):
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
        print(f"Warning: {cls if isinstance(cls, type) else cls.__class__.__name__}.{cls.demo.__name__} Not implemented")
