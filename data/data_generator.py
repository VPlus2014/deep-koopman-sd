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
    """数字转化为带单位的字符串"""
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
    batch_size: int,
    n_steps: int,
    lamb: float,  # 漂移速率
    sigma: float,  # 方差速率
    dt: float,  # 积分步长
    u_space: spaces.Box,
    u_min: np.ndarray,
    u_max: np.ndarray,
    rng: np.random.Generator,
):
    u_n = np.clip([u_space.sample() for _ in range(batch_size)], u_min, u_max)
    dimU = np.shape(u_n)[-1]

    u_mu = 0.5 * (u_min + u_max * np.ones_like(u_n))  # 长期均值
    u_mu[np.isnan(u_mu)] = 0.0

    dt_sqrt = np.sqrt(dt)
    results = [u_n]
    for i in range(n_steps - 1):
        noise = rng.normal(size=(batch_size, dimU)) * (sigma * dt_sqrt)
        u_n = u_n + (u_mu - u_n) * (lamb * dt) + noise
        u_n = np.clip(u_n, u_min, u_max)
        results.append(u_n)
    results = np.asarray(results)

    return np.swapaxes(results, 0, 1)


def _gen_trajs(
    env: DynamicSystemBase,
    us: List[np.ndarray],  # (B,T,dimU)
    Fs: int,
    dt_int: int,
    Ffix: int = None,
    seed: int = None,  # 随机种子
    x0: np.ndarray = None,  # 初始状态
):
    """从单一初始状态生成按不同控制量生成不同轨迹"""
    yss = []
    term = False
    for us_ in us:
        Ts = us_.shape[0]
        y0 = env.reset(seed=seed, x0=x0)
        ys = [y0]
        for k in range(Ts):
            y1, term = env.step(us_[k], dt=dt_int, Fs=Fs, Ffix=Ffix)
            ys.append(y1)
            if term:
                break
        ys = np.asarray(ys)
        yss.append(ys)
    return yss, us


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
    add_zeros_u: bool = True,
    n_workers=None,  # 并行线程/进程数
    use_thread=True,  # 是否使用线程池
    seed: int = None,
):
    """用一组给定环境生成指定时长的合法轨迹"""
    Fs = Fs or 1
    Ffix = Ffix or Fs
    np_rng = np.random.default_rng(seed)
    INT32_MAX = int(np.iinfo(np.int32).max)
    if add_zeros_u:
        assert envs[0].U_space.contains(
            np.zeros_like(envs[0].U_space.low)
        ), "zero control is not valid in env"
    n_envs = len(envs)
    Ys = []  # (N,M,T+1,dimY)
    Us = []  # (N,M,T,dimU)
    from concurrent.futures import Future

    tasks: List[Optional[Future[Tuple[List[np.ndarray], List[np.ndarray]]]]] = [
        None
    ] * n_envs

    max_workers = os.cpu_count()
    n_workers = n_workers or max_workers
    n_workers = min(n_workers, max_workers)
    print(f"max_workers: {n_workers}")
    if use_thread:
        from concurrent.futures import ThreadPoolExecutor

        tpe = ThreadPoolExecutor(max_workers=n_workers)
    else:
        from concurrent.futures import ProcessPoolExecutor

        tpe = ProcessPoolExecutor(max_workers=n_workers)
    n_sub = 0  # 已提交任务数
    pbar = tqdm(total=n_traj)
    while True:
        stop = False
        n_new_ = 0
        # 轮询任务列表
        for ie in range(n_envs):
            tsk = tasks[ie]
            if tsk is not None and tsk.done():
                tasks[ie] = None

                simrst = tsk.result()
                ys, us = simrst
                #
                if all([len(ys_) == Ts + 1 for ys_ in ys]):
                    Ys.append(ys)
                    Us.append(us)

                    n_new_ += 1
                    if len(Ys) >= n_traj:
                        stop = True
                        break

            if tasks[ie] is None and n_sub < n_traj:
                # 启动新任务
                env = envs[ie]
                env_seed = int(np_rng.integers(INT32_MAX))  # 环境随机种子
                env.reset(seed=env_seed)  # 初始化种子&生成初始状态
                x0_ = env._get_state().copy()

                # 随机生成控制量
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
                )  # (1,T,dimU)
                if add_zeros_u:
                    us = np.concatenate([us, np.zeros_like(us)], axis=0)  # (2,T,dimU)

                tsk = tpe.submit(
                    _gen_trajs,
                    env,
                    us,
                    Fs=Fs,
                    dt_int=dt_int,
                    Ffix=Ffix,
                    seed=env_seed,
                    x0=x0_,
                )
                tasks[ie] = tsk
                n_sub += 1

        if n_new_ > 0:
            pbar.update(n_new_)
        if stop:
            break

    tasks = [t for t in tasks if t is not None]
    assert len(tasks) == 0, f"{tasks} task unfinished"
    tpe.shutdown(wait=True)

    Ys = np.asarray(Ys)
    Us = np.asarray(Us)
    return KoopmanData(Ys, Us)


def main():
    seed = None
    nenv = 64  # 最大并行环境数
    n_trajs = 100000  # 总轨迹数
    add_zeros_u = True  # 是否加入零控制量对照组
    n_steps = 50  # 控制输入步数
    Fs = 200  # 采样频率
    dt_int = 1e-3  # 积分步长
    Ffix = 10  # 积分修正间隔
    ou_lambda = 0.5  # 漂移速率
    ou_sigma = 0.1  # 方差速率
    envcls = DOF6Plane
    u_const: np.ndarray = None
    # u0 置 None 表示每条轨迹都独立随机生成控制量，否则所有轨迹在所有时间都沿用这个控制量
    use_thread = False  # 是否使用线程池,否则进程池
    dtp_sim = np.float64  # 仿真数据类型
    dtp_data = np.float32  # 数据类型
    oldversion = False  # 是否使用旧版本数据生成器

    # 自动部分
    env_maker = lambda seed: envcls(seed=seed, dtype=dtp_sim)
    # mini测试环境
    dyna = env_maker(seed)

    def _env_test(env: DynamicSystemBase):
        ts_traj = []
        n_trajs_tst = 2
        pbar = tqdm(total=n_trajs_tst)
        while True:
            t0 = time.time()
            y = env.reset()
            k = 0
            while True:
                u = env.U_space.sample()
                y, term = env.step(u, dt=dt_int, Fs=Fs, Ffix=Ffix)
                k += 1
                trunc = k >= n_steps
                if term or trunc:
                    break
            t2 = time.time()
            if k == n_steps:  # 合法轨迹
                ts_traj.append(t2 - t0)
                pbar.update()
                if len(ts_traj) >= n_trajs_tst:
                    break
        dimY = y.shape[-1]
        dimU = u.shape[-1]
        Dtwall = np.mean(ts_traj)
        Dtsim = k * dt_int * Fs
        secs_est = n_trajs * Dtwall
        if add_zeros_u:
            secs_est *= 2
        _h, _s = divmod(math.ceil(secs_est), 60)
        _h, _m = divmod(_h, 60)
        speed_ratio = Dtsim / max(Dtwall, 1e-6)

        scalar_nbytes = np.dtype(dtp_data).itemsize
        nbytes_est = n_trajs * (dimY + (dimU + dimY) * n_steps) * scalar_nbytes
        if add_zeros_u:
            nbytes_est *= 2
        data_size_est = n2name(nbytes_est)
        print(
            "\n".join(
                [
                    f"{env.__class__.__name__} dimY={dimY} dimU={dimU} n_steps={n_steps}",
                    f"speed ratio={speed_ratio:.3g}x",
                    f"sim {n_trajs} trajs would take at least {_h}:{_m}:{_s}",
                    f"total data size={data_size_est} Bytes",
                ]
            )
        )
        return dimY, dimU

    dimY, dimU = _env_test(dyna)

    sys_name = dyna.__class__.__name__
    #
    use_const_control = u_const is not None
    control_suffix = "cU" if use_const_control else "rU"
    if add_zeros_u:
        control_suffix += "&0"
    #
    Nstr = n2name(n_trajs)  # 数据规模
    data_head = f"{sys_name}Y{dimY}U{dimU}_{control_suffix}_Fs{Fs}_dt{dt_int:4g}_{Nstr}"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    data_outdir = (
        Path(__file__).parent / "raw_data" / f"{data_head}.{timestamp}"
    ).resolve()

    if use_const_control:
        u_const = u_const + np.zeros_like(dyna.X_space.low)
    print(f"data is to be saved to {data_outdir}")

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
            n_trajs,
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
        envs = [env_maker(int(rng.integers(INT32_SUP))) for _ in range(nenv)]
        data = gen_data(
            envs=envs,
            n_traj=n_trajs,
            Ts=n_steps,
            dt_int=dt_int,
            Fs=Fs,
            Ffix=Ffix,
            ou_lambda=ou_lambda,
            ou_sigma=ou_sigma,
            u_min=dyna.U_space.low if not use_const_control else u_const,
            u_max=dyna.U_space.high if not use_const_control else u_const,
            n_workers=nenv,
            use_thread=use_thread,
            add_zeros_u=add_zeros_u,
        )

    data.xs = data.xs.astype(dtp_data)
    data.us = data.us.astype(dtp_data)
    data_trn, data_val = data.train_test_split(0.8)

    data_trn.save(data_outdir / "train")
    data_val.save(data_outdir / "val")
    print(f"data>> {data_outdir}")
    return


if __name__ == "__main__":
    main()
