from pathlib import Path
from typing import Callable, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dataclasses import dataclass
import torch


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

    def fn_inputs(self, dir: Path, prefix=""):
        return f"{dir}/{prefix}_u.npy"

    def fn_traj(self, dir: Path, prefix=""):
        return f"{dir}/{prefix}_x.npy"

    def save(self, dir_name: Path, prefix=""):
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        np.save(self.fn_traj(dir_name, prefix), self.xs)
        np.save(self.fn_inputs(dir_name, prefix), self.us)

    def load(self, dir_name: Path, prefix=""):
        self.xs = np.load(self.fn_traj(dir_name, prefix))
        self.us = np.load(self.fn_inputs(dir_name, prefix))


class Data_Generator:
    """
    Generates driven dynamics using a ODE solver

    Args:
        gradient: function compatable with the scipy's ODE solver,
                  i.e. taking in (x, t, u), and returns dydt
        dt: sampling time
        dim: system's dimension

    Methods:
        generate: runs a set of OU process simulations and returns
                  a Koopman dataclass
    """

    def __init__(self, gradient, dt: float, dim: int, seed=None):
        self.gradient = gradient
        self.dt = dt
        self.dim = dim
        self.data = KoopmanData(None, None)
        self._rng = np.random.default_rng(seed)

    def _ou_generator(self, N, n_steps, params: Tuple[float, float, np.ndarray]):
        theta, sigma, max_u = params
        rng = self._rng

        assert len(max_u) == self.dim

        X_n = (rng.random((N, self.dim)) - 0.5) * 2 * max_u
        mus = (rng.random((N, self.dim)) - 0.5) * 2 * max_u
        results = [X_n]

        for i in range(n_steps - 1):
            noise = sigma * rng.normal(N, self.dim) * np.sqrt(self.dt)
            X_n = X_n + theta * (mus - X_n) * self.dt + noise
            results.append(X_n)
        results = np.array(results)

        for idx, i in enumerate(max_u):
            if i == 0:
                results[:, :, idx] = 0

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
        for i in range(len(us)):
            tspan = [ts[i], ts[i + 1]]
            x0 = odeint(self.gradient, x0, tspan, args=(us[i],))[1]
            traj.append(x0)
            if constraint is not None and constraint(x0):
                is_valid = False
                break
        return np.array(traj), is_valid

    def _apply_constraint(self, result, constraint):
        if constraint is None:
            return True
        a = sum([constraint(i) for i in result])
        if a > 0:
            return False
        else:
            return True

    def _get_initial_cond(self, max_x, constraint):
        while True:
            xs = (self._rng.random([self.dim]) * 2 - 1) * max_x
            if self._apply_constraint([xs], constraint):
                return xs

    def generate(self, N: int, n_steps: int, params: Dict, constraint=None):
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
            constraint: function which returns true if a constraint was violated for the
                        current state
        Returns:
            KoopmanData object containing the generated trajectories and inputs
            xs: (np.ndarray) shape (N, n_steps+1, dim)
            us: (np.ndarray) shape (N, n_steps, dim)
        """

        theta = params.get("theta", 2)
        sigma = params.get("sigma", 1)
        keep_zero = params.get("keep_zero", True)
        max_u = params.get("max_u")
        max_x = params.get("max_x")

        assert len(max_x) == len(max_u) == self.dim

        ts = np.linspace(0, n_steps * self.dt, n_steps + 1)
        trajs = []
        inputs = []
        for i in tqdm(range(N)):
            while True:
                # Generate a random initial condition
                x0_ = self._get_initial_cond(max_x, constraint)  # (dimX,)

                # Generate random inputs
                us1 = self._ou_generator(
                    1, n_steps, [theta, sigma, max_u]
                )  # (1,n_steps,dimU)

                # run input dynamics
                xs1, is_valid = self._simulate(x0_, ts, us1[0], constraint)
                if not is_valid:
                    continue

                # run dynamics with no input
                if keep_zero:
                    us2 = np.zeros_like(us1)
                    xs2, is_valid = self._simulate(x0_, ts, us2[0])
                    if not is_valid:
                        continue

                trajs.append(xs1)
                inputs.append(us1[0])
                if keep_zero:
                    trajs.append(xs2)
                    inputs.append(us2[0])
                break

        xs = np.asarray(trajs)
        us = np.asarray(inputs)
        self.data.us = us
        self.data.xs = xs
        return self.data
