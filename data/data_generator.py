from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import math
import os
import time
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from dynamics import *
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
    meta = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for m, s in meta:
        if n >= m:
            n = float(n / m)
            unit = s
            break
    n = float(n)
    n_ = f"{n:.0f}" if n == int(n) else f"{n:.3g}"
    return f"{n_}{unit}"


def _ou_generator(
    N: int,
    n_steps: int,
    lamb: float,  # 漂移速率
    sigma: float,  # 方差速率
    dt: float,  # 积分步长
    u_space: spaces.Box,
    u_min: np.ndarray,
    u_max: np.ndarray,
    rng: np.random.Generator,
):
    u_n = np.clip([u_space.sample() for _ in range(N)], u_min, u_max)
    dimU = np.shape(u_n)[-1]

    u_mu = 0.5 * (u_min + u_max * np.ones_like(u_n))  # 长期均值
    u_mu[np.isnan(u_mu)] = 0.0

    dt_sqrt = np.sqrt(dt)
    results = [u_n]
    for i in range(n_steps - 1):
        noise = rng.normal(size=(N, dimU)) * (sigma * dt_sqrt)
        u_n = u_n + (u_mu - u_n) * (lamb * dt) + noise
        u_n = np.clip(u_n, u_min, u_max)
        results.append(u_n)
    results = np.asarray(results)

    return np.swapaxes(results, 0, 1)


def gen_traj(
    env: DynamicSystemBase, us: np.ndarray, Fs: int, dt_int: int, Ffix: int = None
):
    term = False
    Ts = us.shape[0]
    y0 = env.reset()
    ys = [y0]
    for k in range(Ts):
        y1, term = env.step(us[k], dt=dt_int, Fs=Fs, Ffix=Ffix)
        ys.append(y1)
        if term:
            break
    ys = np.asarray(ys)
    return ys, us


def gen_data(
    envs: List[DynamicSystemBase],
    n_traj: int,  # 总轨迹数
    Ts: int = 1,  # 总控制步数
    dt_int: Union[int, float] = 1e-3,  # 积分步长
    Fs: int = None,  # 采样频率
    Ffix: int = None,  # 积分修正间隔
    ou_lambda: float = 0.5,  # 漂移速率
    ou_sigma: float = 0.1,  # 方差速率
    u_min: np.ndarray = None,
    u_max: np.ndarray = None,
    n_workers=None,  # 并行线程/进程数
    use_thread=True,  # 是否使用线程池
):
    """用一组给定环境生成指定时长的合法轨迹"""
    Fs = Fs or 1
    Ffix = Ffix or Fs
    n_envs = len(envs)
    Ys = []  # (N, T+1, dimY)
    Us = []  # (N, T, dimU)
    tasks: List[Optional[Future[Tuple[np.ndarray, np.ndarray]]]] = [None] * n_envs

    max_workers = os.cpu_count()
    n_workers = n_workers or max_workers
    n_workers = min(n_workers, max_workers)
    print(f"max_workers: {n_workers}")
    if use_thread:
        tpe = ThreadPoolExecutor(max_workers=n_workers)
    else:
        tpe = ProcessPoolExecutor(max_workers=n_workers)
    n_sub = 0  # 已提交任务数
    pbar = tqdm(total=n_traj)
    while True:
        stop = False
        # 轮询任务列表
        for ie in range(n_envs):
            tsk = tasks[ie]
            if tsk is not None and tsk.done():
                tasks[ie] = None

                simrst = tsk.result()
                ys, us = simrst
                #
                if len(ys) == Ts + 1:
                    Ys.append(ys)
                    Us.append(us)

                    pbar.update()
                    if len(Ys) >= n_traj:
                        stop = True
                        break

            if tasks[ie] is None and n_sub < n_traj:
                # 启动新任务
                env = envs[ie]
                u_min_ = env.U_space.low if u_min is None else u_min
                u_max_ = env.U_space.high if u_max is None else u_max
                us = _ou_generator(
                    1,
                    Ts,
                    lamb=ou_lambda,
                    sigma=ou_sigma,
                    dt=dt_int * Fs,
                    u_space=env.U_space,
                    u_min=u_min_,
                    u_max=u_max_,
                    rng=env._np_random,
                )
                tsk = tpe.submit(gen_traj, envs[ie], us[0], Fs=Fs, dt_int=dt_int)
                tasks[ie] = tsk
                n_sub += 1

        if stop:
            break
        time.sleep(0.001)

    tasks = [t for t in tasks if t is not None and not t.done()]
    assert len(tasks) == 0, f"{tasks} task unfinished"
    tpe.shutdown(wait=True)

    Ys = np.asarray(Ys)
    Us = np.asarray(Us)
    return KoopmanData(Ys, Us)


def main():
    seed = None
    nenv = 64  # 最大并行环境数
    use_thread = False  # 是否使用线程池
    N = 10000
    n_steps = 50  # 控制步数
    Fs = 200  # 采样频率
    dt_int = 1e-3  # 积分步长
    Ffix = 10  # 积分修正间隔
    ou_lambda = 0.5  # 漂移速率
    ou_sigma = 0.1  # 方差速率
    envcls = DOF6PlaneQuat
    u_const: np.ndarray = None
    # u0 置 None 表示每条轨迹都独立随机生成控制量，否则所有轨迹在所有时间都沿用这个控制量
    add_zeros_u = False  # 是否加入零控制量对照组(gym版无效)
    dtp_sim = np.float64  # 仿真数据类型
    dtp_data = np.float32  # 数据类型

    dyna = envcls(seed=seed, dtype=dtp_sim)

    def _demo(env: DynamicSystemBase):
        t0 = time.time()
        env.reset()
        t1 = time.time()
        k = 0
        while True:
            u = env.U_space.sample()
            y, term = env.step(u, dt=dt_int, Fs=Fs, Ffix=Ffix)
            k += 1
            trunc = k >= n_steps
            if term or trunc:
                break
        t2 = time.time()
        dimY = y.shape[-1]
        dimU = u.shape[-1]
        Dtwall = t2 - t0
        Dtsim = k * dt_int * Fs
        dt_per_traj = (t1 - t0) + n_steps * (t2 - t1)
        secs_est = round(N * dt_per_traj)
        _h, _s = divmod(secs_est, 60)
        _h, _m = divmod(_h, 60)
        speed_ratio = Dtsim / max(Dtwall, 1e-6)

        scalar_nbytes = np.dtype(dtp_data).itemsize
        nbytes_est = N * (dimY + (dimU, dimY) * n_steps) * scalar_nbytes
        data_size_est = n2name(nbytes_est)
        print(
            "\n".join(
                [
                    f"{env.__class__.__name__} dimY={dimY} dimU={dimU}",
                    f"{k} steps",
                    f"wall time={Dtwall:.3f}s",
                    f"sim  time={Dtsim:.3f}s",
                    f"speed ratio={speed_ratio:.3g}x",
                    f"therefore sim {N} trajs would take at least {_h}:{_m}:{_s}",
                    f"data size={data_size_est} Bytes",
                ]
            )
        )

    _demo(dyna)
    oldversion = False

    sys_name = dyna.__class__.__name__
    use_const_control = u_const is not None
    control_suffix = "_" + ("cU" if use_const_control else "rU")
    if add_zeros_u and oldversion:
        control_suffix += "&0"

    Nname = n2name(N)
    data_head = f"{sys_name}{control_suffix}_Fs{Fs}_dt{dt_int:4g}_{Nname}"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    data_path = (
        Path(__file__).parent / "raw_data" / f"{data_head}.{timestamp}"
    ).resolve()

    if use_const_control:
        u_const = u_const + np.zeros_like(dyna.X_space.low)
    print(f"data>>{data_path}")

    if oldversion:
        constraint = lambda x: not dyna.X_space.contains(
            x
        )  # !!!数值类型不匹配也算违反约束
        # constraint  = None

        generator = Data_Generator(
            dyna.dynamics,
            dt_int=dt_int,
            fs=Fs,
            seed=seed,
            func_reset=dyna.reset,
        )
        data = generator.generate(
            N,
            n_steps,
            X0_low=dyna.X0_space.low,
            X0_high=dyna.X0_space.high,
            U_low=dyna.U_space.low if not use_const_control else u_const,
            U_high=dyna.U_space.high if not use_const_control else u_const,
            add_zero_u=add_zeros_u and not use_const_control,
            constraint=constraint,
        )
    else:
        rng = np.random.default_rng(seed)
        INT32_SUP = 1 << 31
        nenv = min(nenv, os.cpu_count())  # 限制最大并行环境数
        envs = [
            envcls(seed=int(rng.integers(INT32_SUP)), dtype=dtp_sim)
            for _ in range(nenv)
        ]
        data = gen_data(
            envs=envs,
            n_traj=N,
            Ts=n_steps,
            dt_int=dt_int,
            Fs=Fs,
            Ffix=Ffix,
            ou_lambda=ou_lambda,
            ou_sigma=ou_sigma,
            n_workers=nenv,
            use_thread=use_thread,
        )

    data.xs = data.xs.astype(dtp_data)
    data.us = data.us.astype(dtp_data)
    data_trn, data_val = data.train_test_split(0.8)

    data_trn.save(data_path / "train")
    data_val.save(data_path / "val")


if __name__ == "__main__":
    main()
