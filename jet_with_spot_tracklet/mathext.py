import math
import numpy as np


_PI = math.pi
_2PI = 2 * _PI
_PI_HALF = _PI * 0.5


def modin(x, a=0, m=_2PI):
    """
    a+((x-a) mod m)
    if m=0, return a
    if m>0, y $\in [a,a+m)$
    if m<0, y $\in (a-m,a]$
    """
    y = np.where(m == 0, a, (x - a) % m + a)
    return y


def modrad(x, a=0, m=_2PI):
    return modin(x, a, m)


def rpy_reg(roll_rad: float, pitch_rad: float, yaw_rad: float):
    r"""
    计算 roll pitch yaw 在 $S=[0,2\pi)\times[-\pi/2,\pi/2]\times[0,2\pi)$ 上的最近等价元

    即 $\argmin_{(r,p,y)\in S: R(r,p,y)=R(roll,pitch,yaw)} \|(r,p,y)-(raw,pitch,yaw)\|_2$
    """
    r = modin(roll_rad, 0.0, _2PI)
    p = modin(pitch_rad, -_PI, _2PI)
    y = modin(yaw_rad, 0.0, _2PI)
    if abs(p) > _PI_HALF:
        p = modin(_PI - p, -_PI, _2PI)
        # p = np.clip(p, -_PI_HALF, _PI_HALF)
        r = modin(r + _PI, 0.0, _2PI)
        y = modin(y + _PI, 0.0, _2PI)
    return r, p, y


def rpy_NEDLight2Len(roll_rad: float, pitch_rad: float, yaw_rad: float):
    """将光线系的 NED 姿态 转为相机 amuzi 姿态， 从目标坐标系到两个坐标系的旋转顺序均为 ZYX"""
    r, p, y = rpy_reg(roll_rad, pitch_rad, yaw_rad)
    r_c = modrad(-r, -_PI)
    p_c = _PI_HALF - p  # 转为天顶角, in [0, pi]
    y_c = modrad(-y, -_PI)
    return r_c, p_c, y_c


def rpy_NEDLight2Len_inv(roll_rad: float, pitch_rad: float, yaw_rad: float):
    r_l = modrad(-roll_rad, -_PI)
    p_l = _PI_HALF - pitch_rad  # 转为俯仰角, in [-\pi/2, \pi/2]
    y_l = modrad(-yaw_rad, -_PI)
    return r_l, p_l, y_l


def T_NEDLight_Pic():
    r"""坐标旋转矩阵 T_{LP}, $\Phi_L$=NED 光线系, $\Phi_P$= 右-下-前 图像坐标系;
    $\xi_L = T_{LP} \xi_P$
    """
    return np.asarray(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ],
        float,
    )


def rpy2mat(roll, pitch, yaw, deg=False) -> np.ndarray:
    r"""(\roll, \pitch, \yaw)-> R_{e_3,\yaw}*R_{e_2,\pitch}*R_{e_1,\roll}"""
    _pkg = np
    roll = _pkg.asarray(roll)
    pitch = _pkg.asarray(pitch)
    yaw = _pkg.asarray(yaw)
    _0 = 0 * (roll + pitch + yaw)
    _1 = _0 + 1
    roll = roll + _0
    pitch = pitch + _0
    yaw = yaw + _0
    if deg:
        roll = _pkg.deg2rad(roll)
        pitch = _pkg.deg2rad(pitch)
        yaw = _pkg.deg2rad(yaw)
    c1 = _pkg.cos(roll)
    s1 = _pkg.sin(roll)
    c2 = _pkg.cos(pitch)
    s2 = _pkg.sin(pitch)
    c3 = _pkg.cos(yaw)
    s3 = _pkg.sin(yaw)
    r1 = _pkg.stack(
        [
            _pkg.stack([_1, _0, _0], axis=-1),
            _pkg.stack([_0, c1, -s1], axis=-1),
            _pkg.stack([_0, s1, c1], axis=-1),
        ],
        axis=-2,
    )
    r2 = _pkg.stack(
        [
            _pkg.stack([c2, _0, s2], axis=-1),
            _pkg.stack([_0, _1, _0], axis=-1),
            _pkg.stack([-s2, _0, c2], axis=-1),
        ],
        axis=-2,
    )
    r3 = _pkg.stack(
        [
            _pkg.stack([c3, -s3, _0], axis=-1),
            _pkg.stack([s3, c3, _0], axis=-1),
            _pkg.stack([_0, _0, _1], axis=-1),
        ],
        axis=-2,
    )
    r = r3 @ r2 @ r1
    return r


def rpy2mat_inv(Teb: np.ndarray, roll_ref_rad: float = 0.0, eps=1e-3) -> np.ndarray:
    """rpy2mat 的逆映射，当发生万向节死锁时，需要给出 roll_ref_rad"""
    Teb = np.asarray(Teb, dtype=np.float_)
    assert Teb.shape[-2:] == (3, 3), "expected matrix shape (...,3,3), got {}".format(
        Teb.shape
    )

    s2 = np.clip(-Teb[..., 2, 0], -1.0, 1.0)
    theta = np.arcsin(s2)
    c2s1 = Teb[..., 2, 1]  # R32
    c2c1 = Teb[..., 2, 2]  # R33
    c2s3 = Teb[..., 1, 0]  # R21
    c2c3 = Teb[..., 0, 0]  # R11

    gl = 1 - np.abs(s2) < eps
    _0 = np.zeros_like(theta)
    roll_ref_rad = _0 + roll_ref_rad
    s1 = np.sin(roll_ref_rad)
    c1 = np.cos(roll_ref_rad)
    s2s1 = s1 * s2
    s2c1 = c1 * s2
    R22 = Teb[..., 1, 1]
    R12 = Teb[..., 0, 1]
    R23 = Teb[..., 1, 2]
    R13 = Teb[..., 0, 2]
    s3 = s2s1 * R22 - c1 * R12 + s2c1 * R23 + s1 * R13
    c3 = c1 * R22 + s2s1 * R12 - s1 * R23 + s2c1 * R13

    phi = np.where(gl, roll_ref_rad, np.arctan2(c2s1, c2c1))
    psi = np.where(gl, np.arctan2(s3, c3), np.arctan2(c2s3, c2c3))

    # cy = np.sqrt(Teb[..., 2, 2] * Teb[..., 2, 2] + Teb[..., 1, 2] * Teb[..., 1, 2])
    # euler = np.empty(Teb.shape[:-1], dtype=np.float64)
    # euler[..., 2] = np.where(
    #     condition,
    #     -np.arctan2(Teb[..., 0, 1], Teb[..., 0, 0]),
    #     -np.arctan2(-Teb[..., 1, 0], Teb[..., 1, 1]),
    # )
    # euler[..., 1] = np.where(
    #     condition, -np.arctan2(-Teb[..., 0, 2], cy), -np.arctan2(-Teb[..., 0, 2], cy)
    # )
    # euler[..., 0] = np.where(
    #     condition, -np.arctan2(Teb[..., 1, 2], Teb[..., 2, 2]), 0.0
    # )
    return phi, theta, psi


def rot_demo():
    dr = 1
    dp = 1e-3
    dy = 1
    rs_deg = np.deg2rad(np.arange(-180, 180, dr))
    ps_deg = np.deg2rad(np.linspace(-90, 90, int(180 / dp)))
    ys_deg = np.deg2rad(np.arange(-180, 180, dy))
    batchsize = 10240

    def _proc(rpy_s: list):
        batchsize = len(rpy_s)
        rpy_t = np.asarray(rpy_s)
        assert rpy_t.shape == (batchsize, 3)
        rpy_s.clear()

        teb = rpy2mat(rpy_t[:, 0], rpy_t[:, 1], rpy_t[:, 2])
        rs_p, ps_p, ys_p = rpy2mat_inv(teb, rpy_t[:, 0])

        rpy_p = np.stack([rs_p, ps_p, ys_p], axis=-1)
        assert rpy_p.shape == (batchsize, 3)
        err = modrad(rpy_t - rpy_p, -_PI)
        err = np.linalg.norm(err)
        if err > 1e-6:
            msg = [f"err={err:.6f}"]
            for i in range(len(rpy_t)):
                row = f"[{i}] rpy_t={rpy_t[i]} rpy_p={rpy_p[i]}"
                msg.append(row)
            print("\n".join(msg))

    from tqdm import tqdm

    rpy_s = []
    pbar = tqdm(total=len(rs_deg) * len(ps_deg) * len(ys_deg) // batchsize)
    for r_rad in rs_deg:
        for p_rad in ps_deg:
            for y_rad in ys_deg:
                rpy_s.append([r_rad, p_rad, y_rad])
                if len(rpy_s) >= batchsize:
                    _proc(rpy_s)
                    rpy_s.clear()
                    pbar.update()
    pbar.close()
    if len(rpy_s) > 0:
        _proc(rpy_s)


# def cam_py2mat_inv(teb: np.ndarray):
#     c2 = teb[..., 2, 2]
#     s2 = -teb[..., 2, 0]
#     c3 = teb[..., 1, 1]
#     s3 = -teb[..., 0, 1]
#     pitch: np.ndarray = np.arctan2(s2, c2)
#     yaw: np.ndarray = np.arctan2(s3, c3)
#     return pitch, yaw


if __name__ == "__main__":
    rot_demo()
    pass
