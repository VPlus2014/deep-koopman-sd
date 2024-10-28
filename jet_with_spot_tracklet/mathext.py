import math
import numpy as np

import sys
from pathlib import Path

__DIR_WS = Path(__file__).resolve().parent.parent

sys.path.append(str(__DIR_WS))
from data.dynamics.mathext import *


_PI = math.pi
_2PI = 2 * _PI
_PI_HALF = _PI * 0.5


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


if __name__ == "__main__":
    rot_demo()
    pass
