import math
from typing import List, Union
import numpy as np

_bkbn = np
from numpy import linalg as LA

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


def modrad(x, a=-_PI, m=_2PI):
    return modin(x, a, m)


def moddeg(x, a=-180, m=360):
    return modin(x, a, m)


def R3_wedge(v: np.ndarray):
    r"""$ v_\wedge x = v\times x $"""
    assert v.shape[-1] == 3, "expected last dim to be 3, got {}".format(v.shape)
    v = v.reshape(*v.shape[:-1], 1, 3)
    v1 = v[..., :, 0:1]  # (...,1,1)
    v2 = v[..., :, 1:2]  # (...,1,1)
    v3 = v[..., :, 2:3]  # (...,1,1)
    _0 = np.zeros_like(v1)
    v_times = _bkbn.concatenate(
        [
            _bkbn.concatenate([_0, -v3, v2], axis=-1),
            _bkbn.concatenate([v3, _0, -v1], axis=-1),
            _bkbn.concatenate([-v2, v1, _0], axis=-1),
        ],
        axis=-2,
    )
    return v_times


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
    return _bkbn.asarray(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ],
        _bkbn.float64,
    )


def rpy2mat(
    roll: Union[float, np.ndarray],
    pitch: Union[float, np.ndarray],
    yaw: Union[float, np.ndarray],
    degrees=False,
) -> np.ndarray:
    r"""(\roll, \pitch, \yaw)-> R_{e_3,\yaw}*R_{e_2,\pitch}*R_{e_1,\roll}"""
    # 这套代码兼容 torch
    roll, pitch, yaw = _bkbn.broadcast_arrays(
        *[_bkbn.asarray(x, _bkbn.float64) for x in [roll, pitch, yaw]]
    )

    roll = roll.reshape(*roll.shape, 1, 1)  # (...,1,1)
    pitch = pitch.reshape(*pitch.shape, 1, 1)
    yaw = yaw.reshape(*yaw.shape, 1, 1)
    # assert roll.shape[-1] == 1, "expected last dim to be 1, got {}".format(roll.shape)
    _0 = np.zeros_like(roll)
    _1 = _0 + 1
    if degrees:
        roll = _bkbn.deg2rad(roll)
        pitch = _bkbn.deg2rad(pitch)
        yaw = _bkbn.deg2rad(yaw)
    c1 = _bkbn.cos(roll)
    s1 = _bkbn.sin(roll)
    c2 = _bkbn.cos(pitch)
    s2 = _bkbn.sin(pitch)
    c3 = _bkbn.cos(yaw)
    s3 = _bkbn.sin(yaw)
    cat = _bkbn.concatenate
    r1 = cat(
        [
            cat([_1, _0, _0], axis=-1),
            cat([_0, c1, -s1], axis=-1),
            cat([_0, s1, c1], axis=-1),
        ],
        axis=-2,
    )
    r2 = cat(
        [
            cat([c2, _0, s2], axis=-1),
            cat([_0, _1, _0], axis=-1),
            cat([-s2, _0, c2], axis=-1),
        ],
        axis=-2,
    )
    r3 = cat(
        [
            cat([c3, -s3, _0], axis=-1),
            cat([s3, c3, _0], axis=-1),
            cat([_0, _0, _1], axis=-1),
        ],
        axis=-2,
    )
    r = r3 @ r2 @ r1
    return r


def rpy2mat_inv(Teb: np.ndarray, roll_ref_rad: float = 0.0, eps=1e-6) -> np.ndarray:
    """rpy2mat 的逆映射，当发生万向节死锁时，需要给出 roll_ref_rad"""
    Teb = np.asarray(Teb, dtype=np.float_)
    assert Teb.shape[-2:] == (3, 3), "expected matrix shape (...,3,3), got {}".format(
        Teb.shape
    )

    s2 = _bkbn.clip(-Teb[..., 2, 0], -1.0, 1.0)
    theta = _bkbn.arcsin(s2)
    c2s1 = Teb[..., 2, 1]  # R32
    c2c1 = Teb[..., 2, 2]  # R33
    c2s3 = Teb[..., 1, 0]  # R21
    c2c3 = Teb[..., 0, 0]  # R11

    gl = 1 - _bkbn.abs(s2) < eps
    _0 = _bkbn.zeros_like(theta)
    roll_ref_rad = _0 + roll_ref_rad
    s1 = _bkbn.sin(roll_ref_rad)
    c1 = _bkbn.cos(roll_ref_rad)
    s2s1 = s1 * s2
    s2c1 = c1 * s2
    R22 = Teb[..., 1, 1]
    R12 = Teb[..., 0, 1]
    R23 = Teb[..., 1, 2]
    R13 = Teb[..., 0, 2]
    s3 = s2s1 * R22 - c1 * R12 + s2c1 * R23 + s1 * R13
    c3 = c1 * R22 + s2s1 * R12 - s1 * R23 + s2c1 * R13

    phi = _bkbn.where(gl, roll_ref_rad, _bkbn.arctan2(c2s1, c2c1))
    psi = _bkbn.where(gl, np.arctan2(s3, c3), _bkbn.arctan2(c2s3, c2c3))
    return phi, theta, psi


def rot_demo():
    rng = np.random.default_rng(None)
    shapehead = (64, 50)
    rs = rng.random(size=shapehead) * _2PI
    ps = rng.random(size=shapehead) * _PI
    ys = rng.random(size=shapehead) * _2PI
    vs = rng.random(size=(*shapehead, 3))
    q = rpy2quat(rs, ps, ys)
    v_r1 = quat_rot(q, vs)
    print(v_r1.shape)

    dr = 1e-0
    dp = 1e-0
    dy = 1e-0
    rs_deg = _bkbn.deg2rad(_bkbn.arange(-180, 180, dr))
    ps_deg = _bkbn.deg2rad(_bkbn.linspace(-90, 90, int(180 / dp)))
    ys_deg = _bkbn.deg2rad(_bkbn.arange(-180, 180, dy))
    batchsize = 2048

    err_tol = 1e-12
    err_eul_max = 0.0
    err_T_max = 0.0
    err_TQ_max = 0.0

    def _proc(rpy_s: list):
        nonlocal err_eul_max, err_T_max, err_TQ_max
        batchsize = len(rpy_s)
        rpy_t = _bkbn.asarray(rpy_s)
        assert rpy_t.shape == (batchsize, 3)
        rpy_s.clear()

        Teb = rpy2mat(rpy_t[:, 0], rpy_t[:, 1], rpy_t[:, 2])
        rs_p, ps_p, ys_p = rpy2mat_inv(Teb, rpy_t[:, 0])

        rpy_p = _bkbn.stack([rs_p, ps_p, ys_p], axis=-1)
        assert rpy_p.shape == (batchsize, 3)

        # 欧拉角误差测试
        errs_eul = modrad(rpy_t - rpy_p, -_PI)
        errs_eul = LA.norm(errs_eul, axis=-1)
        errs_eul_btch_ub = errs_eul.max()
        if errs_eul_btch_ub > err_eul_max:
            err_eul_max = errs_eul_btch_ub
        if errs_eul_btch_ub > err_tol:
            msg = []
            for i in range(len(rpy_t)):
                row = f"[{i}] rpy_t={rpy_t[i]} rpy_p={rpy_p[i]}"
                msg.append(row)
            msg.append(f"err={errs_eul_btch_ub:.6g}")
            print("\n".join(msg))

        Qeb = rpy2quat(rpy_t[:, 0], rpy_t[:, 1], rpy_t[:, 2])
        TebQ = quat2mat(Qeb)
        # 矩阵一致性测试
        errs_T = LA.norm(Teb - TebQ, axis=(-2, -1))
        err_T_btch_ub = errs_T.max()
        if err_T_btch_ub > err_tol:
            print(f"|T-T_Q|={err_T_btch_ub:.6g}")
        if err_T_btch_ub > err_T_max:
            err_T_max = err_T_btch_ub

        # 向量旋转测试
        _1s = _bkbn.ones([batchsize, 1])
        e1 = _1s * _e1  # (n,3)
        e2 = _1s * _e2
        e3 = _1s * _e3
        tst_vecs = [e1, e2, e3]
        tst_vecs4mat = [v.reshape(*v.shape, 1) for v in tst_vecs]  # (...,3,1)
        Ys_TebQ = _bkbn.stack([TebQ @ v for v in tst_vecs4mat], axis=-3).squeeze(-1)
        Ys_Qeb = _bkbn.stack([quat_rot(Qeb, v) for v in tst_vecs], axis=-2)
        errs_TQ = LA.norm(Ys_Qeb - Ys_TebQ, axis=-1)
        errs_TQ_btch_ub = errs_TQ.max()
        if errs_TQ_btch_ub > err_tol:
            print(f"|T_Q x-Im(Q*(0,v)*Q^{-1})|={errs_TQ_btch_ub:.6g}")
        if errs_TQ_btch_ub > err_TQ_max:
            err_TQ_max = errs_TQ_btch_ub

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
    print(f"err_eul_max={err_eul_max:.6g}")
    print(f"err_T_max={err_T_max:.6g}")
    print(f"err_TQ_max={err_TQ_max:.6g}")


_e1 = _bkbn.array([1, 0, 0], _bkbn.float_)
_e2 = _bkbn.array([0, 1, 0], _bkbn.float_)
_e3 = _bkbn.array([0, 0, 1], _bkbn.float_)
_I3 = _bkbn.eye(3, dtype=_bkbn.float_)


def quat_prod(p: np.ndarray, q: np.ndarray):
    p = _bkbn.asarray(p, _bkbn.float_)
    q = _bkbn.asarray(q, _bkbn.float_)
    re_p = quat_Re(p)
    im_p = quat_Im(p)
    re_q = quat_Re(q)
    im_q = quat_Im(q)
    re_pq = re_p * re_q - (im_p * im_q).sum(-1, keepdims=True)
    im_pq = re_p * im_q + im_p * re_q + _bkbn.cross(im_p, im_q)
    pq = _bkbn.concatenate([re_pq, im_pq], axis=-1)
    return pq


def rpy2quat(
    roll: Union[float, np.ndarray],
    pitch: Union[float, np.ndarray],
    yaw: Union[float, np.ndarray],
    degrees=False,
):
    r"""
    输入欧拉角，输出四元数 Q=Q_{e3,yaw}*Q_{e2,pitch}*Q_{e1,roll}
    """

    roll, pitch, yaw = _bkbn.broadcast_arrays(
        *[_bkbn.asarray(x, _bkbn.float64) for x in [roll, pitch, yaw]]
    )
    # assert roll.shape[-1] == 1, "expected last dim to be 1, got {}".format(roll.shape)
    r_hf = (roll * 0.5)[..., None]  # (...,1)
    p_hf = (pitch * 0.5)[..., None]
    y_hf = (yaw * 0.5)[..., None]
    if degrees:
        r_hf = _bkbn.deg2rad(r_hf)
        p_hf = _bkbn.deg2rad(p_hf)
        y_hf = _bkbn.deg2rad(y_hf)

    shp1 = [1] * (len(roll.shape) - 1)
    q1 = _bkbn.concatenate(
        [_bkbn.cos(r_hf), _bkbn.sin(r_hf) * _e1.reshape(*shp1, 3)], axis=-1
    )
    q2 = _bkbn.concatenate(
        [_bkbn.cos(p_hf), _bkbn.sin(p_hf) * _e2.reshape(*shp1, 3)], axis=-1
    )
    q3 = _bkbn.concatenate(
        [_bkbn.cos(y_hf), _bkbn.sin(y_hf) * _e3.reshape(*shp1, 3)], axis=-1
    )
    return quat_prod(quat_prod(q3, q2), q1)


def quat2mat(q: np.ndarray, normalize=True):
    """Rv = Q*(0,v)*Q^{-1}"""

    if normalize:
        q = quat_normalize(q)
    _coshf = quat_Re(q)  # (...,1), \cos(\theta/2)
    _coshf = _coshf.reshape(*_coshf.shape, 1)  # (...,1,1)
    _2coshf = _coshf + _coshf  # 2*\cos(\theta/2)
    imQ = quat_Im(q)  # (...,3), \sin(\theta/2) * n, \|n\|_2=1
    imQ_ = imQ.reshape(*imQ.shape, 1)  # (...,3,1)
    imQ_T = imQ_.swapaxes(-1, -2)  # (...,1,3)
    imQ_wedge = R3_wedge(imQ)
    I3 = _I3.reshape(*([1] * (len(q.shape) - 1)), 3, 3)
    m1 = (_2coshf * _coshf) * I3 - I3
    m2 = (2 * imQ_) @ imQ_T
    m3 = _2coshf * imQ_wedge
    r = m1 + m2 + m3
    return r


def quat_Im(q: np.ndarray):
    return q[..., 1:]


def quat_Re(q: np.ndarray):
    return q[..., 0:1]


def quat_conj(q: np.ndarray):
    return np.concatenate([quat_Re(q), -quat_Im(q)], axis=-1)


def quat_norm(q: np.ndarray):
    r"""$\|q\|$"""
    return LA.norm(q, axis=-1, keepdims=True)


def quat_normalize(q: np.ndarray):
    return q / quat_norm(q)


def quat_inv(q: np.ndarray):
    return quat_conj(q) / quat_norm(q)


def quat_from_im(im: np.ndarray):
    r"""[0, im]"""
    return np.concatenate([np.zeros_like(im[..., 0:1]), im], axis=-1)


def quat_rot(q: np.ndarray, v: np.ndarray, normalize=True):
    r"""$Im(q*(0,v)*q^{-1})$"""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    if normalize:
        q = quat_normalize(q)
    h = quat_from_im(v)
    h = quat_prod(q, h)
    h = quat_prod(h, quat_conj(q))
    v = quat_Im(h)
    return v


if __name__ == "__main__":
    rot_demo()
    pass
