"""共线条件方程、旋转矩阵及其一阶偏导（纯 NumPy 实现）。

坐标与角度约定
--------------
* 地面坐标 ``(X, Y, Z)`` 与摄影中心 ``(Xs, Ys, Zs)`` 单位一致（通常为米）。
* 像点坐标 ``(x, y)`` 与相机主距 ``f``、主点偏移 ``(x0, y0)`` 单位一致（通常为毫米）。
* 角元素 ``phi / omega / kappa`` 以**弧度**传入本模块，符合摄影测量右手系约定：

  ``x - x0 = -f * (a1 dX + b1 dY + c1 dZ) / (a3 dX + b3 dY + c3 dZ)``

  ``y - y0 = -f * (a2 dX + b2 dY + c2 dZ) / (a3 dX + b3 dY + c3 dZ)``

  其中 ``dX = X - Xs`` 等，旋转矩阵 ``R = [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]]``。
  当三个角元素均为 0 时 ``R = I``，像平面坐标与地面坐标轴平行。

所有函数都只依赖 NumPy，便于在无图形界面的环境（CI、脚本、测试）中复用。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# 外方位元素顺序：摄影中心坐标 + 三个角元素
PARAM_NAMES = ("Xs", "Ys", "Zs", "phi", "omega", "kappa")


def rotation_matrix(phi: float, omega: float, kappa: float) -> np.ndarray:
    """由三个角元素（弧度）构建 3×3 旋转矩阵。"""
    cp, sp = np.cos(phi), np.sin(phi)
    co, so = np.cos(omega), np.sin(omega)
    ck, sk = np.cos(kappa), np.sin(kappa)

    return np.array(
        [
            [cp * ck - sp * so * sk, co * sk, sp * ck + cp * so * sk],
            [-cp * sk - sp * so * ck, co * ck, -sp * sk + cp * so * ck],
            [-sp * co, -so, cp * co],
        ],
        dtype=float,
    )


def rotation_derivatives(phi: float, omega: float, kappa: float):
    """返回 ``(dR/dphi, dR/domega, dR/dkappa)``，均为弧度制下的解析偏导。"""
    cp, sp = np.cos(phi), np.sin(phi)
    co, so = np.cos(omega), np.sin(omega)
    ck, sk = np.cos(kappa), np.sin(kappa)

    d_phi = np.array(
        [
            [-sp * ck - cp * so * sk, 0.0, cp * ck - sp * so * sk],
            [sp * sk - cp * so * ck, 0.0, -cp * sk - sp * so * ck],
            [-cp * co, 0.0, -sp * co],
        ],
        dtype=float,
    )
    d_omega = np.array(
        [
            [-sp * co * sk, -so * sk, cp * co * sk],
            [-sp * co * ck, -so * ck, cp * co * ck],
            [sp * so, -co, -cp * so],
        ],
        dtype=float,
    )
    d_kappa = np.array(
        [
            [-cp * sk - sp * so * ck, co * ck, -sp * sk + cp * so * ck],
            [-cp * ck + sp * so * sk, -co * sk, -sp * ck - cp * so * sk],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return d_phi, d_omega, d_kappa


def _as_points(ground: Sequence[Sequence[float]], image: Sequence[Sequence[float]]):
    ground_arr = np.asarray(ground, dtype=float)
    image_arr = np.asarray(image, dtype=float)
    if ground_arr.ndim != 2 or ground_arr.shape[1] != 3:
        raise ValueError("地面控制点必须是 n×3 的数组")
    if image_arr.ndim != 2 or image_arr.shape[1] != 2:
        raise ValueError("像点坐标必须是 n×2 的数组")
    if ground_arr.shape[0] != image_arr.shape[0]:
        raise ValueError("地面控制点与像点数量必须一致")
    return ground_arr, image_arr


def camera_coordinates(ground: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """把地面点变换到像空间坐标系，返回 n×3 的 ``(S, U, V)`` 分量。

    ``params`` 依次为 ``Xs, Ys, Zs, phi, omega, kappa``（角度为弧度）。
    """
    params = np.asarray(params, dtype=float)
    rotation = rotation_matrix(params[3], params[4], params[5])
    return (ground - params[:3]) @ rotation.T


def project(
    ground: Sequence[Sequence[float]],
    params: Sequence[float],
    f: float,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """按共线条件方程计算像点坐标（与输入像点同单位），返回 n×2 数组。"""
    ground_arr = np.asarray(ground, dtype=float)
    if ground_arr.ndim != 2 or ground_arr.shape[1] != 3:
        raise ValueError("地面控制点必须是 n×3 的数组")
    cam = camera_coordinates(ground_arr, params)
    v = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = x0 - f * cam[:, 0] / v
        y = y0 - f * cam[:, 1] / v
    return np.column_stack((x, y))


def residuals(
    ground: Sequence[Sequence[float]],
    image: Sequence[Sequence[float]],
    params: Sequence[float],
    f: float,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """观测值减计算值，返回 n×2 的残差矩阵。"""
    ground_arr, image_arr = _as_points(ground, image)
    return image_arr - project(ground_arr, params, f, x0, y0)


def design_matrix(
    ground: Sequence[Sequence[float]],
    params: Sequence[float],
    f: float,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """构建共线方程线性化后的系数矩阵 A（2n×6）。

    每行对应一个观测方程，列顺序为 ``Xs, Ys, Zs, phi, omega, kappa``。
    这里返回的是**计算值对参数的偏导**；线性化形式为 ``A·delta = L``，
    其中 ``L = 观测值 - 计算值``。
    """
    ground_arr = np.asarray(ground, dtype=float)
    if ground_arr.ndim != 2 or ground_arr.shape[1] != 3:
        raise ValueError("地面控制点必须是 n×3 的数组")

    params = np.asarray(params, dtype=float)
    if params.shape != (6,):
        raise ValueError("外方位元素必须包含 6 个分量")
    if not np.isfinite(f) or f <= 0:
        raise ValueError("相机主距必须为正数")

    rotation = rotation_matrix(params[3], params[4], params[5])
    d = ground_arr - params[:3]
    s, u, v = (d @ rotation.T).T
    v_sq = v * v

    if np.any(np.abs(v) < 1e-12):
        # 理论上不会出现，除非点位于投影中心平面；避免除零产生 inf/nan。
        raise ValueError("存在与投影中心同深度的点，无法线性化")

    # 旋转矩阵元素：a1=R00, b1=R01, c1=R02, a2=R10, ...
    a1, b1, c1 = rotation[0]
    a2, b2, c2 = rotation[1]
    a3, b3, c3 = rotation[2]

    n = ground_arr.shape[0]
    jac = np.empty((2 * n, 6), dtype=float)

    # 线元素偏导
    jac[0::2, 0] = f * (a1 * v - a3 * s) / v_sq
    jac[1::2, 0] = f * (a2 * v - a3 * u) / v_sq
    jac[0::2, 1] = f * (b1 * v - b3 * s) / v_sq
    jac[1::2, 1] = f * (b2 * v - b3 * u) / v_sq
    jac[0::2, 2] = f * (c1 * v - c3 * s) / v_sq
    jac[1::2, 2] = f * (c2 * v - c3 * u) / v_sq

    # 角元素偏导
    for column, d_rot in enumerate(rotation_derivatives(params[3], params[4], params[5]), start=3):
        ds, du, dv = (d @ d_rot.T).T
        jac[0::2, column] = -f * (ds * v - s * dv) / v_sq
        jac[1::2, column] = -f * (du * v - u * dv) / v_sq

    return jac
