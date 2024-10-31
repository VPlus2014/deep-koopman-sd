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
    meta = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]
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
    dt: float,  # 积分步长
    u_space: spaces.Box,
    u_min: np.ndarray,
    u_max: np.ndarray,
    rng: np.random.Generator,
    theta: float = None,  # 回归速率,默认 1.0
    sigma: float = None,  # 波动率,默认 1/6.0
):
    r"""
    $$
    U_{t+dt} = \Pi_\mathcal{U}(
        U_t
        +\theta ((u_max+u_min)/2 - U_t) dt
        +\sigma (u_max-u_min) W_t \sqrt(dt)
    ), W_t \sim N(0,1)
    $$
    """
    # assert all(u_min > -np.inf), f"OU u_min must be finite, but got {u_min}"
    # assert all(u_max < np.inf), f"OU u_max must be finite, but got {u_max}"
    u_n = np.asarray([u_space.sample() for _ in range(batch_size)])
    u00: np.ndarray = u_n[0]
    u_min, u_max, u00_ = np.broadcast_arrays(u_min, u_max, u_n[0])
    assert (
        u00_.shape == u00.shape
    ), f"invalid shape u_min:{u_min.shape}, u_max:{u_max.shape}, expected to broadcast to {u00.shape}"
    assert all(
        u_min <= u_max
    ), f"OU expected u_min <= u_max, but got u_min={u_min}, u_max={u_max}"
    dimU = u_n.shape[-1]
    u_n = np.clip(u_n, u_min, u_max)

    u_min = u_min.reshape(1, -1)
    u_max = u_max.reshape(1, -1)
    u_mu = (u_min + u_max) * 0.5  # 长期均值
    u_mu[~np.isfinite(u_mu)] = 0.0

    uspan = u_max - u_min
    uspan[~np.isfinite(uspan)] = 1.0

    if theta is None:
        theta = 1.0
    if sigma is None:
        sigma = 1 / 6.0
    theta_dt = np.reshape(theta * dt, (1, -1))
    sqrt_sig2dt = np.reshape(sigma * math.sqrt(dt), (1, -1)) * uspan
    results = [u_n]
    for i in range(n_steps - 1):
        dX_1 = (u_mu - u_n) * theta_dt
        dX_2 = rng.normal(size=(batch_size, dimU)) * sqrt_sig2dt
        dX = dX_1 + dX_2
        # assert dX.shape == u_n.shape, f"invalid shape dX:{dX.shape}, u_n:{u_n.shape}"
        #
        u_n = u_n + dX
        u_n = np.clip(u_n, u_min, u_max)
        results.append(u_n)
    results = np.asarray(results)
    results = np.swapaxes(results, 0, 1)  # (N,T,dimU)
    return results


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
    env_maker: Callable[[int], DynamicSystemBase],
    n_traj: int,  # 总轨迹数
    Ts: int = 1,  # 总控制步数
    dt_int: Union[int, float] = 1e-3,  # 积分步长
    Fs: int = None,  # 采样频率
    Ffix: int = None,  # 积分修正间隔
    ou_theta: float = None,  # 漂移速率
    ou_sigma: float = None,  # 方差速率
    u_min: np.ndarray = None,
    u_max: np.ndarray = None,
    add_zeros_u: bool = True,
    max_workers: int = None,  # 最大并行数
    env_pool_mode=1,  # 0:串行 1:进程池并行 2:线程池并行
    seed: int = None,
):
    """用一组给定环境生成指定时长的合法轨迹"""
    Fs = Fs or 1
    Ffix = Ffix or Fs
    np_rng = np.random.default_rng(seed)
    INT32_MAX = int(np.iinfo(np.int32).max)
    #
    Ys = []  # (N,M,T+1,dimY)
    Us = []  # (N,M,T,dimU)

    use_async = env_pool_mode != 0
    max_workers_2 = int(os.cpu_count()) if use_async else 1
    nenvs = max_workers or max_workers_2
    nenvs = min(nenvs, max_workers_2)
    envs = [
        env_maker(int(np_rng.integers(INT32_MAX))) for _ in range(nenvs)
    ]  # 串行测试
    n_envs = len(envs)
    print(f"nenvs: {nenvs}")
    assert n_envs > 0, "no envs created"
    if add_zeros_u:
        assert envs[0].U_space.contains(
            np.zeros_like(envs[0].U_space.low)
        ), "zero control is not valid in env"

    from concurrent.futures import Future

    tasks: List[
        Union[
            None,  #
            Tuple[List[np.ndarray], List[np.ndarray]],  #
            Future[Tuple[List[np.ndarray], List[np.ndarray]]],  #
        ]
    ] = [None] * n_envs
    if env_pool_mode == 0:
        pass
    elif env_pool_mode == 1:
        from concurrent.futures import ProcessPoolExecutor

        pe = ProcessPoolExecutor(max_workers=nenvs)
    elif env_pool_mode == 2:
        from concurrent.futures import ThreadPoolExecutor

        pe = ThreadPoolExecutor(max_workers=nenvs)
    else:
        raise ValueError(f"env_pool_mode={env_pool_mode} not supported")
    n_sub = 0  # 已提交任务数
    pbar_sub = tqdm(total=n_traj)
    pbar_done = tqdm(total=n_traj)
    pbar_sub.set_description("trajs submitted\t")
    pbar_done.set_description("trajs finished\t")
    while True:
        stop = False
        n_new_ = 0
        # 轮询任务列表
        for ie in range(n_envs):
            tsk = tasks[ie]
            simrst = None
            if tsk is not None:
                if isinstance(tsk, Future):
                    if tsk.done():
                        simrst = tsk.result()
                else:
                    simrst = tsk
                #
                if simrst is not None:
                    tasks[ie] = None  # 清空已完成任务

                    ys, us = simrst
                    #
                    lens = np.asarray([len(ys_) for ys_ in ys])
                    if all(lens == Ts + 1):
                        Ys.append(ys)
                        Us.append(us)

                        n_new_ += 1
                        pbar_done.update()
                        if len(Ys) >= n_traj:
                            stop = True
                            break
                    # else:
                    #     print(f"obmitted traj lens: {lens}\n")

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
                    theta=ou_theta,
                    sigma=ou_sigma,
                    dt=dt_int * Fs,
                    u_space=env.U_space,
                    u_min=u_min_,
                    u_max=u_max_,
                    rng=env._np_random,
                )  # (1,T,dimU)
                if add_zeros_u:
                    us = np.concatenate([us, np.zeros_like(us)], axis=0)  # (2,T,dimU)

                tsk_args = [env, us, Fs, dt_int, Ffix, env_seed, x0_]
                if use_async:
                    tsk = pe.submit(_gen_trajs, *tsk_args)
                else:
                    tsk = _gen_trajs(*tsk_args)
                tasks[ie] = tsk
                n_sub += 1
                pbar_sub.update()

        # if n_new_ > 0:
        #     pbar.update(n_new_)
        if stop:
            break

    tasks = [t for t in tasks if t is not None]
    assert len(tasks) == 0, f"{tasks} task unfinished"
    pe.shutdown(wait=True)

    Ys = np.asarray(Ys)
    Us = np.asarray(Us)
    return KoopmanData(Ys, Us)


def main():
    seed = None
    max_envs = 64  # 最大并行环境数
    n_trajs = 10_0000  # 总轨迹数
    add_zeros_u = True  # 是否加入零控制量对照组
    n_steps = 50  # 控制输入步数
    Fs = 200  # 采样频率
    dt_int = 1e-3  # 积分步长
    Ffix = 10  # 积分修正间隔
    ou_theta = 0.5  # OU过程 回归速率
    ou_sigma = 0.5 / math.sqrt(Fs * dt_int)  # OU过程 扰动速率
    envcls = DOF6Plane
    u_const: np.ndarray = None
    # u0 置 None 表示每条轨迹都独立随机生成控制量，否则所有轨迹在所有时间都沿用这个控制量
    env_pool_mode = 1  # 0:串行 1:进程池并行 2:线程池并行
    dtp_sim = np.float64  # 仿真数据类型
    dtp_data = np.float32  # 数据类型
    oldversion = False  # 是否使用旧版本数据生成器

    # 自动部分
    def _env_maker(seed: Optional[int] = None):
        return envcls(seed=seed, dtype=dtp_sim)

    # mini测试环境
    dyna = _env_maker(seed)

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
        data = gen_data(
            env_maker=_env_maker,
            n_traj=n_trajs,
            Ts=n_steps,
            dt_int=dt_int,
            Fs=Fs,
            Ffix=Ffix,
            ou_theta=ou_theta,
            ou_sigma=ou_sigma,
            u_min=dyna.U_space.low if not use_const_control else u_const,
            u_max=dyna.U_space.high if not use_const_control else u_const,
            max_workers=max_envs,
            env_pool_mode=env_pool_mode,
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
