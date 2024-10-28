# 函数描述：生成三个频段飞机轨迹数据集，带干扰弹
# 2024年3月24日，生成姿态生成核心代码
# 输入给定的姿态
# 输出对应的飞机图片，以及对应的关键点标签信息
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union
import cv2
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import json

PACK_DIR = Path(__file__).resolve().parent

from mathext import *


class Tracklet_generation:
    def __init__(self, raw_data_path: str, multi=1):
        """
        初始化函数，设置计算器的初始值。

        参数:
            initial_value (int, 可选): 计算器的初始值，默认为0。
            multi 代表两种数据集，一种是单波段的，一种是多波段的，分别是0和1表示
        """
        # 相机内参矩阵示例
        self.depth = 50
        fx = 20000  # 假设焦距为512/27 pix/m
        fy = 20000  # 假设焦距为512/27 pix/m
        if multi == 0:
            cx, cy = 330, 258  # 假设图像中心为原点（根据实际情况调整）#329.3  258.8
        else:
            cx, cy = 318, 256
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K_inv = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]])

        """初始点位置"""
        # raw_data_path = D:\\funding\\No17\\middle\\data\\MB1
        self.raw_data_path = Path(raw_data_path).resolve()
        raw = (
            self.raw_data_path / "label_initial_frame" / "pitch0_yaw0.json"
        )  #'D:\\funding\\No17\\middle\\data\\labels\\kpimgs\\俯仰角0_方位角0.json'
        json_path = raw

        # 读取并解析JSON文件，指定编码为UTF-8
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 按照label排序并提取关键点
        shapes = sorted(data["shapes"], key=lambda x: int(x["label"]))
        self.points_init = [shape["points"][0] for shape in shapes]
        self.folder = "D:\\funding\\No17\\middle\\data\\SortedImages\\pitch"
        # self.points_init = self.rotate_only_keypoints(self.points_init, np.array([330,259]), -90)

    def euler_to_rot_matrix(self, yaw, pitch, roll):
        # 将角度转换为弧度
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        # 计算旋转矩阵
        R_pitch = np.array(
            [
                [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                [0, 1, 0],
                [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
            ]
        )

        # 绕z轴旋转的矩阵（方位角）
        R_yaw = np.array(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix
        R = np.dot(R_pitch, R_yaw)
        return R

    def euler_to_rot_matrix_multi(self, yaw, pitch, roll):
        # 将角度转换为弧度
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        # 计算旋转矩阵
        # R_pitch = np.array([
        #     [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        #     [0, 1, 0],
        #     [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        # ])
        R_pitch = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0, np.sin(pitch_rad), np.cos(pitch_rad)],
            ]
        )
        # 绕z轴旋转的矩阵（方位角）
        R_yaw = np.array(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix
        R = np.dot(R_pitch, R_yaw)
        return R

    def point_rotataion(self, points_init, pitch, yaw):
        def project_2d_to_3d(points_2d, K_inv, depth):
            """将2D点投影到3D空间中，假设所有点在同一深度"""
            # 转换为齐次坐标
            kk = np.ones((points_2d.shape[0], 1))
            points_hom = np.hstack([points_2d, kk])
            # 通过相机内参逆矩阵投影到3D
            points_3d = (K_inv @ points_hom.T).T
            # points_3d[:, -1] = depth*points_3d[:, -1]
            return points_3d[:, :3]

        def rotate_points_3d(points_3d, R):
            """应用3D旋转"""
            return (R @ points_3d.T).T

        def project_3d_to_2d(points_3d, K):
            """将3D点投影回2D图像平面"""
            # 将3D点转换为齐次坐标
            # 使用K矩阵将3D点投影到2D，注意这里直接使用3D点的前3维进行投影
            points_2d = points_3d[:, :] / points_3d[:, [2]]
            points_2d = (
                K @ points_2d.T
            ).T  # 注意这里改为使用 points_3d 而非 points_hom
            # 将齐次2D坐标转换为普通2D坐标
            return points_2d[:, :2]

        # 2D点坐标示例
        points_2d = np.array(points_init)
        points_3d = project_2d_to_3d(points_2d, self.K_inv, self.depth)
        rod = self.depth + np.array(
            [-0.01, 0.0, 0.0, 0.100, -0.01, 0.100, 0.012, 0.012]
        )
        # given the deth information
        for i in range(3):
            points_3d[:, i] = points_3d[:, i] * rod
        # 绕着 z轴旋转
        R = self.euler_to_rot_matrix(-yaw, 0, 0)
        points_3d_rotated = rotate_points_3d(points_3d, R)

        points_3d_rotated[:, -1] = points_3d_rotated[:, -1] - self.depth
        points_3d_rotated[:, 0] = points_3d_rotated[:, 0] + 0.02
        R = self.euler_to_rot_matrix(0, -pitch, 0)
        points_3d_rotated = rotate_points_3d(points_3d_rotated, R)
        points_3d_rotated[:, -1] = points_3d_rotated[:, -1] + self.depth
        points_3d_rotated[:, 0] = points_3d_rotated[:, 0] - 0.02
        points_2d_projected = project_3d_to_2d(points_3d_rotated, self.K)
        return points_2d_projected

    def point_rotataion_multi(self, points_init, pitch, yaw):
        def project_2d_to_3d(points_2d, K_inv, depth):
            """将2D点投影到3D空间中，假设所有点在同一深度"""
            # 转换为齐次坐标
            kk = np.ones((points_2d.shape[0], 1))
            points_hom = np.hstack([points_2d, kk])
            # 通过相机内参逆矩阵投影到3D
            points_3d = (K_inv @ points_hom.T).T
            # points_3d[:, -1] = depth*points_3d[:, -1]
            return points_3d[:, :3]

        def rotate_points_3d(points_3d, R):
            """应用3D旋转"""
            return (R @ points_3d.T).T

        def project_3d_to_2d(points_3d, K):
            """将3D点投影回2D图像平面"""
            # 将3D点转换为齐次坐标
            # 使用K矩阵将3D点投影到2D，注意这里直接使用3D点的前3维进行投影
            points_2d = points_3d[:, :] / points_3d[:, [2]]
            points_2d = (
                K @ points_2d.T
            ).T  # 注意这里改为使用 points_3d 而非 points_hom
            # 将齐次2D坐标转换为普通2D坐标
            return points_2d[:, :2]

        # 2D点坐标示例
        points_2d = np.array(points_init)
        points_3d = project_2d_to_3d(points_2d, self.K_inv, self.depth)

        rod = self.depth + np.array(
            [-0.02, -0.02, -0.02, 0.05, -0.01, 0.05, -0.02, -0.02]
        )
        # rod = self.depth
        # given the deth information
        for i in range(3):
            points_3d[:, i] = points_3d[:, i] * rod
        # 绕着 z轴旋转
        R = self.euler_to_rot_matrix(yaw, 0, 0)
        points_3d_rotated = rotate_points_3d(points_3d, R)
        points_3d_rotated[:, -1] = points_3d_rotated[:, -1] - self.depth
        points_3d_rotated[:, 0] = points_3d_rotated[:, 0] + 0.02
        R = self.euler_to_rot_matrix_multi(0, pitch, 0)
        points_3d_rotated = rotate_points_3d(points_3d_rotated, R)
        points_3d_rotated[:, -1] = points_3d_rotated[:, -1] + self.depth
        points_3d_rotated[:, 0] = points_3d_rotated[:, 0] - 0.02
        points_2d_projected = project_3d_to_2d(points_3d_rotated, self.K)
        return points_2d_projected

    def rotate_only_keypoints(self, keypoints, ksize, angle):
        # Calculate the rotation matrix manually since we don't have an image
        image_center = (int(ksize[0]), int(ksize[1]))
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, scale=1)
        # 获取图像尺寸
        # Translate keypoints to origin based on the center point
        rotated_keypoints = []
        for point in keypoints:
            # 将关键点转换为齐次坐标
            original_point = np.array([point[0], point[1], 1])
            # 使用旋转矩阵计算旋转后的位置
            rotated_point = rotation_matrix @ original_point
            rotated_keypoints.append((rotated_point[0], rotated_point[1]))
        return rotated_keypoints

    def roll_image_and_keypoints(self, image, keypoints, angle):
        # 获取图像尺寸
        height, width = image.shape[:2]
        # 计算图像中心点
        image_center = (width / 2, height / 2)
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, scale=1)
        # 旋转图像
        rotated_image = cv2.warpAffine(
            image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        # 旋转关键点
        rotated_keypoints = []
        for point in keypoints:
            # 将关键点转换为齐次坐标
            original_point = np.array([point[0], point[1], 1])
            # 使用旋转矩阵计算旋转后的位置
            rotated_point = rotation_matrix @ original_point
            rotated_keypoints.append((rotated_point[0], rotated_point[1]))
        return rotated_image, rotated_keypoints

    def find_even(self, n):
        # 找到小于等于n的最大整数
        lower_int = math.floor(n)
        # 找到大于等于n的最小整数
        upper_int = math.ceil(n)

        # 确保lower_int是偶数，如果是奇数，则减1
        if lower_int % 2 != 0:
            lower_int -= 1

        # 确保upper_int是偶数，如果是奇数，则加1
        if upper_int % 2 != 0:
            upper_int += 1

        # 比较n与lower_int和upper_int的距离，返回更近的偶数
        if n - lower_int <= upper_int - n:
            return lower_int
        else:
            return upper_int

    def pitch_yaw_row_img(self, points_init, pitch, yaw, roll):
        """rotation"""
        pitch, yaw, roll = 90 + pitch, 180 + yaw, -90 + roll
        pitch, yaw, roll = (
            self.find_even(pitch),
            self.find_even(yaw),
            self.find_even(roll),
        )
        """边界限定"""
        if pitch > 180:
            pitch = pitch - 180
        if pitch < 0:
            pitch = pitch + 180
        if yaw > 358:
            yaw = yaw - 360
        raw = "D:\\funding\\No17\\middle\\data\\SortedImages\\pitch" + str(pitch)
        folder_path = raw + "\pitch" + str(pitch) + "_yaw" + str(yaw) + ".png"
        image_path = folder_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points_rotated = self.point_rotataion(points_init, pitch, yaw)
        image, keypoints = self.roll_image_and_keypoints(image, points_rotated, roll)
        return image, keypoints

    def pitch_yaw_row_img_multispec(self, points_init, pitch, yaw, roll):
        """rotation"""
        pitch, yaw, roll = 90 + pitch, 180 - yaw, roll
        pitch, yaw, roll = int(pitch), int(yaw), int(roll)
        """边界限定"""
        if pitch > 180:
            pitch = pitch - 180
        if pitch < 0:
            pitch = pitch + 180
        if yaw > 359:
            yaw = yaw - 360

        # base = 'D:\\funding\\No17\\middle\\data\\SortedImages\\pitch'
        def read_img(base: Path, pitch: int, yaw: int):
            key_pitch = f"pitch{pitch}"
            key_yaw = f"yaw{yaw}"
            img_fn = base / key_pitch / f"{key_pitch}_{key_yaw}.png"
            # print(img_fn)
            image = cv2.imread(str(img_fn))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            points_rotated = self.point_rotataion_multi(points_init, pitch, yaw)

            image, keypoints = self.roll_image_and_keypoints(
                image, points_rotated, roll
            )
            return image, keypoints

        swir_base = self.raw_data_path / "SWIR"
        mwir_base = self.raw_data_path / "MWIR"
        lwir_base = self.raw_data_path / "LWIR"
        swir_img, keypoints = read_img(swir_base, pitch, yaw)
        mwir_img, _ = read_img(mwir_base, pitch, yaw)
        lwir_img, _ = read_img(lwir_base, pitch, yaw)
        return swir_img, mwir_img, lwir_img, keypoints

    def histogram_matching(self, src_img: np.ndarray, ref_img: np.ndarray):
        """
        将源图像(src_img)的直方图匹配到参考图像(ref_img)的直方图。
        """
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        # 计算源图像和参考图像的直方图及累积分布函数(CDF)
        src_hist, _ = np.histogram(src_img.flatten(), 256, [0, 256])
        src_cdf = src_hist.cumsum()
        src_cdf_normalized = src_cdf / src_cdf.max()
        ref_hist, _ = np.histogram(ref_img.flatten(), 256, [0, 256])
        ref_cdf = ref_hist.cumsum()
        ref_cdf_normalized = ref_cdf / ref_cdf.max()
        # 构建查找表
        lookup_table = np.zeros(256)
        ref_idx = 0
        for src_idx in range(256):
            while (
                ref_cdf_normalized[ref_idx] < src_cdf_normalized[src_idx]
                and ref_idx < 255
            ):
                ref_idx += 1
            lookup_table[src_idx] = ref_idx
        # 应用查找表
        matched_img = cv2.LUT(src_img.astype("uint8"), lookup_table.astype("uint8"))
        return matched_img

    def pose_img_kyp_generation_multispec(self, attitude):
        # 【0，1，0】先进行旋转，之后算出相机系下的旋转角度，还有相机系下的旋转向量
        # ，再根据旋转向量，得到对应的pitch和yaw的值，最后显示图片，并且旋转图片
        # 假设飞机的pitch,row,yaw分别是0，2，2
        def roll_image(image: np.ndarray, angle_deg: float):
            # 获取图像尺寸
            height, width = image.shape[:2]
            # 计算图像中心点
            image_center = ((width - 1) / 2, (height - 1) / 2)
            # 计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(image_center, angle_deg, scale=1)
            # 旋转图像
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            return rotated_image

        def rotate_vector(vector, angle_degrees, axis):
            rotation = R.from_euler(axis, angle_degrees, degrees=True)
            rotated_vector = rotation.apply(vector)
            return rotated_vector

        def rotate_vector_to_make_x_zero(x, vector, z):
            # 计算需要旋转的角度（弧度），使向量的 x 分量为 0
            angle_radians = np.arctan2(vector[1], vector[0])
            angle_degrees = np.degrees(angle_radians)
            # 创建绕 z 轴旋转的 Rotation 对象
            rotation = R.from_euler("z", 90 - angle_degrees, degrees=True)
            # 应用旋转
            rotated_vector = rotation.apply(vector)
            rotated_vectorx = rotation.apply(x)
            rotated_vectorz = rotation.apply(z)
            return rotated_vectorx, rotated_vector, rotated_vectorz, 90 - angle_degrees

        x = [1, 0, 0]
        y = [0, 1, 0]
        z = [0, 0, 1]

        rotation = R.from_euler("xyz", attitude, degrees=True)
        rotation = rotation.inv()
        x = rotation.apply(x)
        y = rotation.apply(y)
        z = rotation.apply(z)
        # 获取相机旋转，补偿滚转
        rotated_vectoryx, rotated_vectory, rotated_vectoryz, rotation_angle = (
            rotate_vector_to_make_x_zero(x, y, z)
        )

        def calculate_rotation_angles(a, b, c):
            rotation = R.align_vectors([a, b, c], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])[0]
            rotation = rotation.inv()
            rotation_matrix = rotation.as_matrix()
            euler_angles = rotation.as_euler("xyz", degrees=True)
            angle_x, angle_y, angle_z = euler_angles
            return angle_x, angle_y

        angle_x, angle_y = calculate_rotation_angles(
            rotated_vectoryx, rotated_vectory, rotated_vectoryz
        )
        pitch = angle_x
        yaw = angle_y
        roll = rotation_angle
        """消除超过90度的误差问题"""
        if pitch < -90 or pitch > 90:
            pitch = pitch - np.sign(pitch) * 180
            yaw = np.sign(yaw) * 180 - yaw
        # image, kpts = self.pitch_yaw_row_img(self.points_init, pitch, yaw, roll)
        swir_img, mwir_img, lwir_img, kpts = self.pitch_yaw_row_img_multispec(
            self.points_init, pitch, yaw, roll
        )
        swir_2_mwir = self.histogram_matching(swir_img, mwir_img)
        lwir_2_mwir = self.histogram_matching(lwir_img, mwir_img)
        return swir_img, mwir_img, lwir_img, swir_2_mwir, lwir_2_mwir, kpts


# 2024年3月24日，生成姿态生成核心代码

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
"""左转、右、上、下、直飞等不同方向轨迹部分"""
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
import math
import os
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import random


def quat_linspace(start: np.ndarray, end: np.ndarray, num: int):
    pass


_e1 = np.array([1, 0, 0], np.float_)
_e2 = np.array([0, 1, 0], np.float_)
_e3 = np.array([0, 0, 1], np.float_)


def quat_prod(p: np.ndarray, q: np.ndarray):
    p = np.asarray(p, np.float_)
    q = np.asarray(q, np.float_)
    re_p = p[..., 0:1]
    im_p = p[..., 1:]
    re_q = q[..., 0:1]
    im_q = q[..., 1:]
    re_pq = re_p * re_q - (im_p * im_q).sum(-1, keepdims=True)
    im_pq = re_p * im_q + im_p * re_q + np.cross(im_p, im_q)
    pq = np.concatenate([re_pq, im_pq], axis=-1)
    return pq


def rpy2quat(roll: float, pitch: float, yaw: float, degrees=True):
    r"""
    输入欧拉角，输出四元数 Q=Q_{e1,roll}*Q_{e2,pitch}*Q_{e3,yaw}
    """
    r_hf = np.array(roll, np.float_, copy=False, ndmin=2) * 0.5
    p_hf = np.array(pitch, np.float_, copy=False, ndmin=2) * 0.5
    y_hf = np.array(yaw, np.float_, copy=False, ndmin=2) * 0.5
    if degrees:
        r_hf = np.deg2rad(r_hf)
        p_hf = np.deg2rad(p_hf)
        y_hf = np.deg2rad(y_hf)
    q1 = np.concatenate([np.cos(r_hf), np.sin(r_hf) * _e1], axis=-1)
    q2 = np.concatenate([np.cos(p_hf), np.sin(p_hf) * _e1], axis=-1)
    q3 = np.concatenate([np.cos(y_hf), np.sin(y_hf) * _e1], axis=-1)
    return quat_prod(q1, quat_prod(q2, q3))


def left_trajectory_with_camera_rotation():
    # 生成飞机轨迹和姿态
    max_roll = rng.uniform(30, 60)
    start_pitch = rng.uniform(-20, 20)
    end_pitch = rng.uniform(-20, 20)
    final_yaw = rng.uniform(50, 110)
    camera_rotation_roll = rng.uniform(-0, 0, size=1)
    camera_rotation_yaw = rng.uniform(-20, 20, size=1)
    camera_rotation_pitch = rng.uniform(-20, 20, size=1)
    camera_rotation_angles = [
        camera_rotation_pitch[0],
        camera_rotation_yaw[0],
        camera_rotation_roll[0],
    ]
    # print(camera_rotation_angles)
    ts = np.linspace(0, 10, 100)
    rolls = np.sin(np.pi * ts / 10) * max_roll
    yaws = np.linspace(0, final_yaw, len(ts))
    pitchs = np.linspace(start_pitch, end_pitch, len(ts))

    delta_x = np.cos(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_y = np.sin(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_z = np.sin(np.deg2rad(pitchs))

    x = -np.cumsum(delta_x)
    y = -np.cumsum(delta_y)
    z = np.cumsum(delta_z)
    # 初始化存储每个点相对姿态的数组
    relative_attitudes = np.zeros((len(ts), 3))

    def camera_relative_attitude_quaternion(pitchs, yaws, rolls, cr):
        relative_attitude = [pitchs + cr[0], yaws + cr[1], rolls + cr[2]]
        return relative_attitude

    # 计算轨迹上每个点的飞机相对于相机的姿态信息
    for index in range(len(ts)):
        relative_attitudes[index] = camera_relative_attitude_quaternion(
            pitchs[index], yaws[index], rolls[index], camera_rotation_angles
        )
    return x, y, z, relative_attitudes
    # return relative_attitudes


def right_trajectory_with_camera_rotation():
    # 生成飞机轨迹和姿态
    max_roll = rng.uniform(-60, -30)  # 右转滚转角为负值
    start_pitch = rng.uniform(-20, 20)
    end_pitch = rng.uniform(-20, 20)
    final_yaw = rng.uniform(-110, -50)  # 右转偏航角为负值
    camera_rotation_roll = rng.uniform(-0, 0, size=1)
    camera_rotation_yaw = rng.uniform(-85, 85, size=1)
    camera_rotation_pitch = rng.uniform(-20, 20, size=1)
    camera_rotation_angles = [
        camera_rotation_pitch[0],
        camera_rotation_yaw[0],
        camera_rotation_roll[0],
    ]

    t = np.linspace(0, 10, 100)
    rolls = np.sin(np.pi * t / 10) * max_roll
    yaws = np.linspace(0, final_yaw, len(t))  # 从0到一个负值
    pitchs = np.linspace(start_pitch, end_pitch, len(t))

    delta_x = np.cos(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_y = np.sin(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_z = np.sin(np.deg2rad(pitchs))

    x = np.cumsum(delta_x)
    y = np.cumsum(delta_y)
    z = np.cumsum(delta_z)

    # 初始化存储每个点相对姿态的数组
    relative_attitudes = np.zeros((len(t), 3))

    def camera_relative_attitude_quaternion(pitchs, yaws, rolls, cr):
        relative_attitude = [pitchs + cr[0], yaws + cr[1], rolls + cr[2]]
        return relative_attitude

    # 计算轨迹上每个点的飞机相对于相机的姿态信息
    for index in range(len(t)):
        relative_attitudes[index] = camera_relative_attitude_quaternion(
            pitchs[index], yaws[index], rolls[index], camera_rotation_angles
        )
    return x, y, z, relative_attitudes
    # return relative_attitudes


def down_trajectory_with_camera_rotation():
    # 生成飞机轨迹和姿态
    # 生成飞机轨迹和姿态
    max_roll = rng.uniform(-20, 20)  # 轻微滚转以模拟真实飞行的微小不稳定
    start_pitch = rng.uniform(0, 15)  # 开始时向上的俯仰角
    max_pitch = rng.uniform(25, 55)  # 最大爬升角
    end_pitch = rng.uniform(0, 15)  # 最终平飞的俯仰角设为0
    start_yaw = rng.uniform(-20, 20)
    end_yaw = rng.uniform(-20, 20)
    camera_rotation_roll = rng.uniform(-0, 0, size=1)
    camera_rotation_yaw = rng.uniform(-20, 20, size=1)
    camera_rotation_pitch = rng.uniform(-20, 20, size=1)
    camera_rotation_angles = [
        camera_rotation_pitch[0],
        camera_rotation_yaw[0],
        camera_rotation_roll[0],
    ]

    t = np.linspace(0, 10, 100)
    rolls = np.ones_like(t) * max_roll

    # 使用正弦函数模拟平滑的俯仰角变化
    # pitchs = (np.sin(np.pi * t / max(t)) * (max_pitch - start_pitch)) + start_pitch
    pitchs = np.sin(np.pi * t / 10) * max_pitch
    # 确保俯仰角平滑过渡到平飞
    pitchs[-10:] = np.linspace(pitchs[-10], end_pitch, 10)

    yaws = np.linspace(
        start_yaw, end_yaw, len(t)
    )  # 假设在整个飞行过程中飞机有轻微的偏航角变化

    delta_x = np.sin(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_y = np.cos(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_z = np.sin(np.deg2rad(pitchs))

    x = np.cumsum(delta_x)
    y = np.cumsum(delta_y)
    z = np.cumsum(delta_z)

    # 初始化存储每个点相对姿态的数组
    relative_attitudes = np.zeros((len(t), 3))

    def camera_relative_attitude_quaternion(pitchs, yaws, rolls, cr):
        relative_attitude = [pitchs + cr[0], yaws + cr[1], rolls + cr[2]]
        return relative_attitude

    # 计算轨迹上每个点的飞机相对于相机的姿态信息
    for index in range(len(t)):
        relative_attitudes[index] = camera_relative_attitude_quaternion(
            pitchs[index], yaws[index], rolls[index], camera_rotation_angles
        )
    return x, y, z, relative_attitudes
    # return relative_attitudes


def up_trajectory_with_camera_rotation():
    # 生成飞机轨迹和姿态
    # 生成飞机轨迹和姿态
    max_roll = rng.uniform(-20, 20)  # 轻微滚转以模拟真实飞行的微小不稳定
    max_pitch = rng.uniform(-55, -25)  # 最大爬升角
    start_yaw = rng.uniform(-10, 10)
    end_yaw = rng.uniform(-10, 10)
    camera_rotation_roll = rng.uniform(-0, 0, size=1)
    camera_rotation_yaw = rng.uniform(-10, 10, size=1)
    camera_rotation_pitch = rng.uniform(-20, 20, size=1)
    camera_rotation_angles = [
        camera_rotation_pitch[0],
        camera_rotation_yaw[0],
        camera_rotation_roll[0],
    ]

    t = np.linspace(0, 10, 100)
    rolls = np.ones_like(t) * max_roll

    # 使用正弦函数模拟平滑的俯仰角变化
    # pitchs = (np.sin(np.pi * t / max(t)) * (max_pitch - start_pitch)) + start_pitch
    pitchs = np.sin(np.pi * t / 10) * max_pitch

    yaws = np.linspace(
        start_yaw, end_yaw, len(t)
    )  # 假设在整个飞行过程中飞机有轻微的偏航角变化

    delta_x = np.sin(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_y = np.cos(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_z = np.sin(np.deg2rad(pitchs))

    x = np.cumsum(delta_x)
    y = np.cumsum(delta_y)
    z = np.cumsum(delta_z)

    # 初始化存储每个点相对姿态的数组
    relative_attitudes = np.zeros((len(t), 3))

    def camera_relative_attitude_quaternion(pitchs, yaws, rolls, cr):
        relative_attitude = [pitchs + cr[0], yaws + cr[1], rolls + cr[2]]
        return relative_attitude

    # 计算轨迹上每个点的飞机相对于相机的姿态信息
    for index in range(len(t)):
        relative_attitudes[index] = camera_relative_attitude_quaternion(
            pitchs[index], yaws[index], rolls[index], camera_rotation_angles
        )
    return x, y, z, relative_attitudes
    # return relative_attitudes


def straight_trajectory_with_camera_rotation():
    # 生成飞机轨迹和姿态

    start_yaw = rng.uniform(-20, 20)
    end_yaw = rng.uniform(-20, 20)

    start_roll = rng.uniform(-20, 20)
    end_roll = rng.uniform(-20, 20)

    start_pitch = rng.uniform(-20, 20)
    end_pitch = rng.uniform(-20, 20)
    camera_rotation_roll = rng.uniform(-0, 0, size=1)
    camera_rotation_yaw = rng.uniform(-15, 15, size=1)
    camera_rotation_pitch = rng.uniform(-15, 15, size=1)
    camera_rotation_angles = [
        camera_rotation_pitch[0],
        camera_rotation_yaw[0],
        camera_rotation_roll[0],
    ]

    t = np.linspace(0, 10, 100)

    # 使用正弦函数模拟平滑的俯仰角变化
    # pitchs = (np.sin(np.pi * t / max(t)) * (max_pitch - start_pitch)) + start_pitch
    pitchs = np.linspace(start_pitch, end_pitch, len(t))
    rolls = np.linspace(start_roll, end_roll, len(t))
    yaws = np.linspace(
        start_yaw, end_yaw, len(t)
    )  # 假设在整个飞行过程中飞机有轻微的偏航角变化

    delta_x = np.cos(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_y = np.sin(np.deg2rad(yaws)) * np.cos(np.deg2rad(pitchs))
    delta_z = np.sin(np.deg2rad(pitchs))

    x = np.cumsum(delta_x)
    y = np.cumsum(delta_y)
    z = np.cumsum(delta_z)

    # 初始化存储每个点相对姿态的数组
    relative_attitudes = np.zeros((len(t), 3))

    def camera_relative_attitude_quaternion(pitchs, yaws, rolls, cr):
        relative_attitude = [pitchs + cr[0], yaws + cr[1], rolls + cr[2]]
        return relative_attitude

    # 计算轨迹上每个点的飞机相对于相机的姿态信息
    for index in range(len(t)):
        relative_attitudes[index] = camera_relative_attitude_quaternion(
            pitchs[index], yaws[index], rolls[index], camera_rotation_angles
        )
    return x, y, z, relative_attitudes
    # return relative_attitudes


def resize_keypoints(
    keypoints: Union[List[Tuple[float, float]], np.ndarray],
    original_width: int,
    original_height: int,
    new_width: int,
    new_height: int,
):
    xr = (new_width - 1) / (original_width - 1)
    yr = (new_height - 1) / (original_height - 1)
    scaled_xy = [(x * xr, y * yr) for x, y in keypoints]
    scaled_xy = np.asarray(scaled_xy, np.float_)
    assert all(scaled_xy[:, 0] <= new_width - 1), "scaled y > new_width-1"
    assert all(scaled_xy[:, 1] <= new_height - 1), "scaled y > new_height-1"
    return scaled_xy


# 给出干扰弹生成相关函数
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
"""给出干扰弹生成相关函数"""
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


def simulate_3d_flare_motion(
    z_range: Tuple[float, float], total_time=20, resolution=100
):
    t = np.linspace(0, total_time, resolution)
    vx_initial = rng.uniform(-5, 5)
    vz_initial = rng.uniform(z_range[0], z_range[1])  # [60,100]
    ax = rng.uniform(-0.2, 0.2, resolution)
    az = np.full(resolution, -9.8)

    x, z = [0], [0]
    for i in range(1, resolution):
        t_delta = t[i] - t[i - 1]
        vx = vx_initial + np.sum(ax[:i]) * t_delta
        vz = vz_initial + np.sum(az[:i]) * t_delta
        x.append(x[-1] + vx * t_delta)
        z.append(z[-1] + vz * t_delta)

    return np.asarray(x, np.float_), np.asarray(z, np.float_)


def transform_to_image_coordinates(
    xs: np.ndarray, zs: np.ndarray, image_size, projection_range
):
    bx, rx = projection_range["x"]
    by, ry = projection_range["z"]

    # 计算原始轨迹的范围
    x_min, x_max = np.min(xs), np.max(xs)
    z_min, z_max = np.min(zs), np.max(zs)
    # 计算缩放比例
    scale_x = (xs - x_min) * (rx / (x_max - x_min))
    scale_z = (zs - z_min) * (ry / (z_max - z_min))

    # 应用缩放和平移变换
    x_transformed = bx + scale_x - scale_x[0]
    z_transformed = by + scale_z - scale_z[0]
    # print(z_transformed)
    # 图像坐标系Y轴翻转
    z_transformed = image_size[1] - z_transformed

    return x_transformed, z_transformed


def size_change(index, total_points, max_size):
    # 定义四个阶段的分界点
    up_phase_end = total_points // 4  # 尺寸增大结束时的索引
    down_phase_start = 3 * total_points // 4  # 尺寸减小开始时的索引

    # 上升期：尺寸快速增大至最大值
    if index <= up_phase_end:
        size = 1 + (index / up_phase_end) * (
            max_size - 1
        )  # 从1变化到max_size，尺寸增加更快
    # 维持期：尺寸保持最大值
    elif index <= down_phase_start:
        size = max_size  # 尺寸维持在最大值
    # 下降期：尺寸线性减小回到原始大小
    else:
        size = max_size - (
            (index - down_phase_start) / (total_points - down_phase_start)
        ) * (
            max_size - 1
        )  # 从max_size变化到1
    return size


def create_blurred_circle(diameter, blur_radius: int, circle_color, x, y):
    """Create a blurred circle on a transparent background."""
    # 创建一个透明背景图像
    size = (diameter + 2 * blur_radius, diameter + 2 * blur_radius)
    img = Image.new("RGBA", size, (71, 71, 71, 71))
    draw = ImageDraw.Draw(img)

    # 绘制实心圆
    draw.ellipse(
        (blur_radius, blur_radius, diameter + blur_radius, diameter + blur_radius),
        fill=circle_color,
    )

    # 应用高斯模糊
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    circle_array = np.array(img)
    circle_diameter = circle_array.shape[0]
    left = int(x) - circle_diameter / 2
    right = int(x) + circle_diameter / 2
    bottom = int(y) - circle_diameter / 2
    top = int(y) + circle_diameter / 2
    extent = [left, right, bottom, top]
    return img, extent


def create_blurred_circle_Img(spot_size: int):
    """Create a blurred circle on a transparent background."""
    # 创建一个透明背景图像
    diameter = int(spot_size) + 1
    blur_radius = int(diameter * 0.15) + 0
    circle_color = (255, 255, 255, 255)  # RGBA

    size = (diameter + 2 * blur_radius, diameter + 2 * blur_radius)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # 绘制实心圆
    draw.ellipse(
        (blur_radius, blur_radius, diameter + blur_radius, diameter + blur_radius),
        fill=circle_color,
    )
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return img


def simulate_radiation_and_projection_corrected(
    total_time=10,
    resolution=100,
    max_spot_size=1,
    z_range: Tuple[float, float] = [60, 100],
):
    # 辐射模拟
    t = np.linspace(0, total_time, resolution)
    radiation = np.sin(np.pi * t / total_time) ** 2  # 辐射强度随时间的变化
    # 三维运动轨迹
    x, y, z = simulate_3d_flare_motion(
        z_range, total_time=total_time, resolution=resolution
    )
    # 光斑大小与辐射强度成比例，同时确保大小在一个合理范围内
    x = -x * 10
    z = z * 0.01
    spot_sizes = radiation * max_spot_size
    return x, z, spot_sizes


def spot_tracklets(
    x_scaled_init: float,
    z_scaled_init: float,
    max_spot_size: int,
    spot_sx_size: List[int],
    z_range: Tuple[float, float],
):
    image_size = [640, 512]
    projection_range = {
        "x": [
            x_scaled_init,
            rng.uniform(spot_sx_size[0], spot_sx_size[1]),
        ],  # x轴投影到图像中的[100, 540]范围内
        "z": [z_scaled_init, rng.uniform(spot_sx_size[2], spot_sx_size[3])],
    }  # z轴投影到图像中的[100, 412]范围内
    # 生成干扰弹轨迹
    xs, zs = simulate_3d_flare_motion(z_range)
    # 转换到图像坐标系
    x_spot, z_spot = transform_to_image_coordinates(
        xs, zs, image_size, projection_range
    )
    spot_sizes = [
        size_change(i, len(x_spot), max_spot_size) for i in range(len(x_spot))
    ]
    return x_spot, z_spot, spot_sizes


def resize_pix(
    xys: Union[List[Tuple[float, float]], np.ndarray],
    original_width: int,
    original_height: int,
    new_width: int,
    new_height: int,
):
    xys = np.array(xys, np.float_)
    xr = (new_width - 1) / (original_width - 1)
    yr = (new_height - 1) / (original_height - 1)
    xys[..., 0] *= xr
    xys[..., 1] *= yr
    return xys


def _proc_frame(
    i_frame: int,
    img_f: Image.Image,
    img_b: Image.Image,
    rand_resize: int,
    src_keypoints: Union[List[Tuple[float, float]], np.ndarray],
    src_W: int,
    src_H: int,
    src_x: float,
    src_y: float,
    rpy: Tuple[float, float, float],
    dir_img: Path,
    dir_json: Path,
    create_blurred_circle_Img: Callable[[int], Image.Image] = create_blurred_circle_Img,
    render_spots: bool = True,
    spots_all: List[Dict] = [],
    render_keypoints: bool = True,
    verbose: bool = True,
):
    pic_head = f"{i_frame}"
    new_size = (img_f.width // rand_resize, img_f.height // rand_resize)
    ic_dst_x = (new_size[0] - 1) // 2
    ic_dst_y = (new_size[1] - 1) // 2
    kpts_dst = resize_pix(src_keypoints, src_W, src_H, new_size[0], new_size[1])

    # 平移关键点以匹配在背景图像中的新位置
    adjusted_keypoints = [
        [
            kpt_x - ic_dst_x + src_x,
            kpt_y - ic_dst_y + src_y,
        ]
        for kpt_x, kpt_y in kpts_dst
    ]

    resized_image = img_f.resize(new_size)
    # 将缩放后的图像放置在背景图上的定位置
    img_b.paste(
        resized_image,
        [
            int(src_x - ic_dst_x),
            int(src_y - ic_dst_y),
        ],
    )
    # 添加干扰弹图片
    # 是否添加干扰弹
    if render_spots:
        for spot_info in spots_all:
            if i_frame > spot_info["idx"]:
                x_spot, z_spot, spot_sizes = (
                    spot_info["x_spot"],
                    spot_info["z_spot"],
                    spot_info["spot_sizes"],
                )
                circle_spot = create_blurred_circle_Img(spot_sizes[i_frame])
                img_b.paste(
                    circle_spot,
                    [int(x_spot[i_frame]), int(z_spot[i_frame] - 10)],
                    circle_spot,
                )
    # 绘制关键点,测试时启用，否则把关键点存起来就行
    if render_keypoints:
        img_b = np.array(img_b)
        for x, y in adjusted_keypoints:
            img_b = cv2.circle(
                img_b,
                (int(x), int(y)),
                radius=3,
                color=(255, 0, 0),
                thickness=-1,
            )
        img_b = Image.fromarray(img_b)
    # 根据xyz缩放图片，并把图像贴到背景中
    fn_pic = dir_img / f"{pic_head}.png"
    img_b.save(fn_pic)
    if verbose:
        print("img>>", fn_pic)
    #
    fn_json = dir_json / f"{pic_head}.json"
    with open(fn_json, "w") as json_file:
        json.dump(
            {
                "keypoints": adjusted_keypoints,
                "pitch": rpy[0],
                "roll": rpy[1],
                "yaw": rpy[2],
            },
            json_file,
            indent=4,
        )
    if verbose:
        print("json>>", fn_json)
    return


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
"""main function 参数配置部分"""
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
seed = None  # 设置随机种子
rng = np.random.default_rng(seed)
tracklet_num = 1  # 每个 (动作,波段) 组合生成的轨迹数量
src_W = 640  # 源图片尺寸
src_H = 512  # 源图片尺寸
modes = [1, 2, 3, 4, 5]  # 一次生成动作数，mode = [1,2,3,4,5] # 左右上下直
TRACK_SIZE = (
    8  # 轨迹在图像中的幅度大小，比如8代表，左转或者右转在图像中转完了width/8个幅度
)
# 背景图像的大小和颜色
background_size = (src_W, src_H)  #
background_colors = {
    "swir": (101, 101, 101),
    "mwir": (101, 101, 101),
    "lwir": (101, 101, 101),
}
# 图像中目标随机尺寸范围,数值越大，目标越小
MIN_FIGHTER_SIZE, MAX_FIGHTER_SIZE = 2, 5
render_keypoints = False  # 如果要检查生成的关键点是否正确绘到图上，可以设置为True
# 干扰弹相关参数设置
render_spots = False  # 是否添加干扰弹
spot_max_size = 20  # 干扰弹最大尺寸
spot_idxs = [
    0,
    1,
    3,
    5,
    10,
    12,
    15,
    20,
    25,
    30,
]  # 在第几帧开始出现干扰弹, 3代表从第三帧开始生成一个新的干扰弹，以此类推
spot_sx_size = [
    50,
    100,
    100,
    200,
]  # 干扰弹的随机抛散在图像中的约束范围，前两个值为x方向约束range，即随机生成的值在【0】和【1】之间产生

Domain_AD = False  #

# 比如随机出一个75，也就是干扰弹曲线最大值和初始值之间的像素距离为75像素
# 针对每个mode，为了增加对sopt的遮挡，每个轨迹给出了独特的size
# 输入输出跟目标的目录
dir_output = PACK_DIR / "data" / "demo"  # 输出目录
raw_data_path = PACK_DIR / "MB1"  # 原始图片存放根目录
pose_generation = Tracklet_generation(raw_data_path)
SAVE_Origianl = True
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
"""main function 循环执行部分"""
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


def main():
    # 加载图像变换模型
    for mode in modes:
        for i_traj in range(tracklet_num):
            # 随机左转生成轨迹
            if mode == 1:
                rst = left_trajectory_with_camera_rotation()
                mode_str = "left"
                z_range = [60, 100]
            elif mode == 2:
                rst = right_trajectory_with_camera_rotation()
                mode_str = "right"
                z_range = [60, 100]
            elif mode == 4:
                rst = down_trajectory_with_camera_rotation()
                mode_str = "down"
                z_range = [20, 40]

            elif mode == 3:
                rst = up_trajectory_with_camera_rotation()
                mode_str = "up"
                z_range = [80, 120]
            elif mode == 5:
                rst = straight_trajectory_with_camera_rotation()
                z_range = [40, 80]
                mode_str = "straight"
            else:
                raise ValueError(f"Invalid mode {mode}")

            rand_resize = rng.integers(MIN_FIGHTER_SIZE, MAX_FIGHTER_SIZE)
            xs, ys, zs, relative_attitudes = rst
            pitchs, yaws, rolls = (
                relative_attitudes[:, 0],
                relative_attitudes[:, 1],
                relative_attitudes[:, 2],
            )
            folder1 = dir_output / mode_str
            print(f"当前正在生成 mode={mode_str}, i_trajectory={i_traj}")
            # 根据roll pitch 和 yaw 进行转换
            folder2 = "trace" + str(i_traj)
            save_dir_pic = folder1 / folder2
            save_dir_pic.mkdir(parents=True, exist_ok=True)
            # multi spectral
            mults = ["swir", "mwir", "lwir"]
            for multi in mults:
                tmp = save_dir_pic / multi
                tmp.mkdir(parents=True, exist_ok=True)
            save_dir_json = Path(str(folder1) + "_json") / folder2
            save_dir_json.mkdir(parents=True, exist_ok=True)
            # scale the track
            # 将轨迹缩放到图像大小
            xmin: float = np.min(xs)
            xmax: float = np.max(xs)
            zmin: float = np.min(zs)
            zmax: float = np.max(zs)
            zspan = zmax - zmin
            xspan = xmax - xmin
            assert xspan > 0, "xspan should be positive"
            assert zspan > 0, "zspan should be positive"
            xs_scaled: np.ndarray = (xs - xmin) / xspan * src_W
            zs_scaled: np.ndarray = (zs - zmin) / zspan * src_H
            xz_fac = xspan / zspan
            if xz_fac < 1:
                xs_scaled = (xs_scaled * xz_fac) // TRACK_SIZE
                zs_scaled = zs_scaled // TRACK_SIZE
            else:
                xs_scaled = xs_scaled // TRACK_SIZE
                zs_scaled = (zs_scaled / xz_fac) // TRACK_SIZE

            # 图像中心像素坐标
            ic_src_x = (src_W - 1) / 2
            ic_src_y = (src_H - 1) / 2

            # 平移轨迹，使初始点位于图像中心
            x_scaled_centered = xs_scaled + (ic_src_x - xs_scaled[0])
            z_scaled_centered = zs_scaled + (ic_src_y - zs_scaled[0])

            # 确保平移后的坐标不会超出图像边界
            xs_scaled = np.clip(x_scaled_centered, 0, src_W - 1)
            zs_scaled = np.clip(z_scaled_centered, 0, src_H - 1)

            # 根据飞机的中心位置x_scaled和z_scaled添加干扰弹，默认干扰弹初始时刻开始释放
            # image_size = [640, 512]
            # # 指定投影范围
            # projection_range = {
            #     'x': [x_scaled[0], x_scaled[0]+100],  # x轴投影到图像中的[100, 540]范围内
            #     'z': [z_scaled[0], z_scaled[0]+200] }  # z轴投影到图像中的[100, 412]范围内
            # # 生成干扰弹轨迹
            # x, z = simulate_3d_flare_motion()
            # # 转换到图像坐标系
            # x_spot, z_spot = transform_to_image_coordinates(x, z, image_size, projection_range)
            # spot_sizes = [size_change(i, len(x_img), max_spot_size) for i in range(len(x_img))]
            # 构建多个干扰弹的轨迹
            spots_all: List[Dict[str, Any]] = []
            if render_spots:
                for i_frame in spot_idxs:
                    xs_spot, zs_spot, spot_sizes = spot_tracklets(
                        xs_scaled[i_frame],
                        zs_scaled[i_frame],
                        spot_max_size,
                        spot_sx_size,
                        z_range,
                    )
                    spots_all.append(
                        {
                            "idx": i_frame,
                            "x_spot": xs_spot,
                            "z_spot": zs_spot,
                            "spot_sizes": spot_sizes,
                        }
                    )
            for i_frame in range(len(pitchs)):
                attitude = [pitchs[i_frame], yaws[i_frame], rolls[i_frame]]
                image_s, image_m, image_l, swir_2_mwir, lwir_2_mwir, keypoints = (
                    pose_generation.pose_img_kyp_generation_multispec(attitude)
                )
                if Domain_AD:
                    multi_images = {
                        "swir": swir_2_mwir,
                        "mwir": image_m,
                        "lwir": lwir_2_mwir,
                    }
                    multi_original = {"swir": image_s, "mwir": image_m, "lwir": image_l}
                else:
                    multi_images = {"swir": image_s, "mwir": image_m, "lwir": image_l}

                for specral in mults:
                    image = multi_images[specral]
                    image = Image.fromarray(image)
                    # 创建背景图像
                    background_color = background_colors[specral]
                    background = Image.new("RGB", background_size, background_color)

                    _proc_frame(
                        i_frame=i_frame,
                        img_b=background,
                        img_f=image,
                        rand_resize=rand_resize,
                        src_keypoints=keypoints,
                        src_W=src_W,
                        src_H=src_H,
                        src_x=xs_scaled[i_frame],
                        src_y=zs_scaled[i_frame],
                        rpy=(pitchs[i_frame], yaws[i_frame], rolls[i_frame]),
                        dir_img=save_dir_pic / specral,
                        dir_json=save_dir_json,
                        spots_all=spots_all,
                        render_spots=render_spots,
                        render_keypoints=render_keypoints,
                        verbose=False,
                    )


if __name__ == "__main__":
    main()
