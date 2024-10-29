import math
import os
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from dynamics import Pendulum, DOF6PlaneQuat
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


def pendulum_too_fast(x):
    theta, dtheta = x
    return


def duffing_dynamics(y, t, u):
    x1, x2 = y
    dydt = [x2, x1 - x1**3 + u[1]]
    return dydt


def toy_dynamics(y, t, u):
    x1, x2 = y
    dydt = [-0.05 * x1, -1.0 * (x2 - x1**2) + u[1] ** 2]
    return dydt


def n2name(n: int):
    unit = ""
    meta = [(1e9, "B"), (1e6, "M"), (1e3, "K")]
    for m, s in meta:
        if n >= m:
            n = float(n / m)
            unit = s
            break
    n = float(n)
    n_ = f"{n:.0f}" if n == int(n) else f"{n:.3g}"
    return f"{n_}{unit}"


def main():
    seed = 123
    N = 10000
    n_steps = 50
    dt_int = 1e-2
    fs = 20
    dyna = DOF6PlaneQuat(seed=seed)
    # dyna.demo()
    sys_name = dyna.__class__.__name__
    u0 = None  # 置 None 表示每条轨迹都独立随机生成控制量，否则所有轨迹在所有时间都沿用这个控制量
    add_zeros_u = True  # 是否加入零控制量对照组
    constraint = lambda x: not dyna.X_space.contains(x)  # !!!数值类型不匹配也算违反约束
    # constraint  = None

    generator = Data_Generator(
        dyna.dynamics,
        dt_int=dt_int,
        fs=fs,
        seed=seed,
    )

    use_const_control = u0 is not None
    control_suffix = "_" + ("cU" if use_const_control else "rU")
    if add_zeros_u:
        control_suffix += "&0"

    Nname = n2name(N)
    data_head = f"{sys_name}{control_suffix}_Fs{fs}_dt{dt_int:4g}_{Nname}"

    data_path = (Path(__file__).parent / "raw_data" / data_head).resolve()

    if use_const_control:
        u0 = u0 + np.zeros_like(dyna.X_space.low)
    print(f"data>>{data_path}")
    data = generator.generate(
        N,
        n_steps,
        X0_low=dyna.X0_space.low,
        X0_high=dyna.X0_space.high,
        U_low=dyna.U_space.low if not use_const_control else u0,
        U_high=dyna.U_space.high if not use_const_control else u0,
        add_zero_u=add_zeros_u and not use_const_control,
        constraint=constraint,
    )
    data_trn, data_val = data.train_test_split(0.8)

    data_trn.save(data_path / "train")
    data_val.save(data_path / "val")


if __name__ == "__main__":
    main()
