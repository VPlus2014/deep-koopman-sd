import os

import numpy as np
import matplotlib.pyplot as plt
from data_helper_fns import *


def dynamics(y, t, u):
    x1, x2 = y
    dydt = [x2, -np.sin(x1) + u[1]]
    return dydt


def is_constraint_violated(x):
    x0, x1 = x
    # potential = 0.5 * x1**2 - np.cos(x0)
    potential = 0.5 * x1**2 - 0.5 * x0**2 + 0.25 * x0**4
    return potential > 0.99


def pendulum_dynamics(y, t, u):
    x1, x2 = y
    dydt = [x2, -np.sin(x1) + u[1]]
    return dydt


def duffing_dynamics(y, t, u):
    x1, x2 = y
    dydt = [x2, x1 - x1**3 + u[1]]
    return dydt


def toy_dynamics(y, t, u):
    x1, x2 = y
    dydt = [-0.05 * x1, -1.0 * (x2 - x1**2) + u[1] ** 2]
    return dydt


def main():
    N = 25000
    n_steps = 50
    dt = 0.01
    generator = Data_Generator(pendulum_dynamics, dt, 2)

    params = {
        "theta": 2,
        "sigma": 0,
        "keep_zero": True,
        "max_u": [0, 1],
        "max_x": [3.14, 2.5],
    }

    sysname = "pend"
    data_path = (Path(__file__).parent / "raw_data").resolve()

    data = generator.generate(N, n_steps, params, is_constraint_violated)
    data_trn, data_val = data.train_test_split(0.8)
    data_trn.save(data_path, f"{sysname}_train")
    data_val.save(data_path, f"{sysname}_val")


if __name__ == "__main__":
    main()
