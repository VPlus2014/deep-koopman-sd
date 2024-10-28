import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dataclasses import dataclass
from gymnasium import spaces


def convex_combination(x1, x2, alpha) -> np.ndarray:
    r"""$(1-\alpha) * x1 + \alpha * x2$"""
    x1 = np.asarray(x1)
    return x1 + alpha * (x2 - x1)


@dataclass
class KoopmanData:
    xs: np.ndarray
    us: np.ndarray

    def shuffle(self):
        p = np.random.permutation(len(self.xs))
        return self.xs[p], self.us[p]

    def train_test_split(self, ratio=0.7):
        """-> train_data, test_data"""
        xs, us = self.shuffle()
        idx = int(len(xs) * ratio)
        train_x, train_u = xs[:idx], us[:idx]
        test_x, test_u = xs[idx:], us[idx:]
        return (
            KoopmanData(train_x, train_u),
            KoopmanData(test_x, test_u),
        )

    def fn_inputs(self, fn_head: Path):
        return f"{fn_head}_u.npy"

    def fn_traj(self, fn_head: Path):
        return f"{fn_head}_x.npy"

    def save(self, fn_head: Path):
        Path(fn_head).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.fn_traj(fn_head), self.xs)
        np.save(self.fn_inputs(fn_head), self.us)

    def load(self, dir_name: Path):
        self.xs = np.load(self.fn_traj(dir_name))
        self.us = np.load(self.fn_inputs(dir_name))


class Data_Generator:
    """
    Generates driven dynamics using a ODE solver

    Args:
        gradient: function compatable with the scipy's ODE solver,
                  i.e. taking in (x, t, u), and returns dydt
        dt_int: integration time step
        fs: sampling frequency
        dim: system's dimension

    Methods:
        generate: runs a set of OU process simulations and returns
                  a Koopman dataclass
    """

    def __init__(
        self,
        gradient: Callable[[float, np.ndarray, Any], np.ndarray],
        dt_int: float,
        fs: int = 1,
        dtype=np.float_,
        seed=None,
    ):
        self.gradient = gradient
        self._dt_int = dt_int
        self._fs = fs
        assert 0 < fs and math.isfinite(fs), "Sampling frequency should be positive"
        self._data = KoopmanData(None, None)
        self._rng = np.random.default_rng(seed)
        self.dtype = dtype

    def _ou_generator(
        self,
        N: int,
        n_steps: int,
        lamb: float,
        sigma: float,
        min_u: np.ndarray,
        max_u: np.ndarray,
        U_space: spaces.Box,
    ):
        rng = self._rng
        u_n = np.clip([U_space.sample() for _ in range(N)], min_u, max_u)
        dimU = np.shape(u_n)[-1]
        u_mu = 0.5 * (min_u + max_u * np.ones_like(u_n))  # 长期均值
        u_mu[np.isnan(u_mu)] = 0.0

        dt = self._fs * self._dt_int
        dt_sqrt = np.sqrt(dt)
        results = [u_n]
        for i in range(n_steps - 1):
            noise = rng.normal(size=(N, dimU)) * (sigma * dt_sqrt)
            u_n = u_n + (u_mu - u_n) * (lamb * dt) + noise
            u_n = np.clip(u_n, min_u, max_u)
            results.append(u_n)
        results = np.asarray(results)

        return np.swapaxes(results, 0, 1)

    def _simulate(
        self,
        x0: np.ndarray,
        ts: List[float],
        us: np.ndarray,
        constraint: Callable = None,
    ):
        """"""
        traj = [x0]
        is_valid = True
        dti = self._dt_int
        for i in range(len(us)):
            tspan = [ts[i], ts[i + 1]]
            ys = odeint(self.gradient, x0, tspan, args=(us[i],), tfirst=True, hmax=dti)
            if self._all_subject_to(ys, constraint):
                x0 = ys[-1]
                traj.append(x0)
            else:
                is_valid = False
                break
        return np.array(traj), is_valid

    def _all_subject_to(
        self, xs: List[np.ndarray], constraint: Callable[[np.ndarray], float]
    ):
        if constraint is None:
            return True
        for x in xs:
            if constraint(x) > 0:
                return False
        return True

    def _get_initial_cond(
        self,
        X0_space: spaces.Box,
        X0_low: np.ndarray,
        X0_high: np.ndarray,
        constraint: Callable[[np.ndarray], float],
    ):
        while True:
            x0 = np.clip(X0_space.sample(), X0_low, X0_high)
            if self._all_subject_to([x0], constraint):
                return x0

    def generate(
        self,
        N: int,  # 轨迹数
        n_steps: int,  # 每条轨迹的迭代步数
        X0_low: np.ndarray,  # 初始状态下界
        X0_high: np.ndarray,  # 初始状态上界
        U_low: np.ndarray,  # 输入下界
        U_high: np.ndarray,  # 输入上界
        ou_theta=2,
        ou_sigma=1,
        add_zero_u=True,  # 以零控制量为对照组生成另一条轨迹
        constraint: Callable[[np.ndarray], float] = None,
        seed=None,
    ):
        """
        Args:
            N: (int) number of trajectories simulated
            n_steps: (int) number of time-steps for each simulation
            params: (dict) configuration dictionary
                    "theta": (float), parameter for OU process
                    "sigma": (float), parameter for OU process
                    "keep_zero": (bool) half of trajectories generated will be autonomous
                    "max_u": (float) maximum input magnitude
                    "max_x"" (float) maximum state magnitude
            constraint: function which returns true if a constraint was violated for the\
                    current state
        Returns:
            KoopmanData object containing the generated trajectories and inputs
            xs: (np.ndarray) shape (N, n_steps+1, dim)
            us: (np.ndarray) shape (N, n_steps, dim)
        """
        dtype = self.dtype
        X0_low = np.asarray(X0_low, dtype=dtype)
        X0_high = np.asarray(X0_high, dtype=dtype)
        U_low = np.asarray(U_low, dtype=dtype)
        X0_space = spaces.Box(X0_low, X0_high, dtype=dtype, seed=seed)
        U_space = spaces.Box(U_low, U_high, dtype=dtype, seed=seed)
        assert all((U_low <= 0) & (U_high >= 0)), "Input space should contain zeros"

        ts = np.linspace(0, n_steps * (self._fs * self._dt_int), n_steps + 1)
        trajs = []
        inputs = []
        for i in tqdm(range(N)):
            while True:
                # Generate a random initial condition
                x0_ = self._get_initial_cond(
                    X0_space, X0_low, X0_high, constraint
                )  # (dimX,)

                # Generate random inputs
                us1 = self._ou_generator(
                    1,
                    n_steps,
                    lamb=ou_theta,
                    sigma=ou_sigma,
                    min_u=U_low,
                    max_u=U_high,
                    U_space=U_space,
                )  # (1,n_steps,dimU)

                # run input dynamics
                xs1, is_valid = self._simulate(x0_, ts, us1[0], constraint)
                if not is_valid:
                    continue

                # run dynamics with no input
                if add_zero_u:
                    us2 = np.zeros_like(us1)
                    xs2, is_valid = self._simulate(x0_, ts, us2[0])
                    if not is_valid:
                        continue

                trajs.append(xs1)
                inputs.append(us1[0])
                if add_zero_u:
                    trajs.append(xs2)
                    inputs.append(us2[0])
                break

        xs = np.asarray(trajs)
        us = np.asarray(inputs)
        self._data.us = us
        self._data.xs = xs
        return self._data
