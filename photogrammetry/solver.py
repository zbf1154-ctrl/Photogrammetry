"""外方位元素解算核心：稳健初值 + Levenberg–Marquardt 光束法平差。

与旧版相比的主要改进：

* 解析偏导代替近似系数，任意倾角下均严密；
* 用二维相似变换估计初值（比例尺、旋角、摄影中心平面位置），
  避免旧版“重心上方 1000 m”造成的发散；
* Levenberg–Marquardt 阻尼 + 多组初值，兼顾收敛速度与稳健性；
* 自动识别像点 y 轴向下（像素坐标系）导致的镜像；
* 输出残差、单位权中误差、参数精度、条件数与几何质量告警。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .model import PARAM_NAMES, design_matrix, project

DEG = math.pi / 180.0

#: 至少需要的控制点数量（每个点提供 2 个方程，未知数 6 个）
MIN_POINTS = 3


class SolveError(ValueError):
    """解算条件不满足或解算失败时抛出的异常。"""


@dataclass
class Solution:
    """一次外方位元素解算的完整结果。"""

    xs: float
    ys: float
    zs: float
    phi: float  # 度
    omega: float  # 度
    kappa: float  # 度
    f: float
    x0: float = 0.0
    y0: float = 0.0
    n_points: int = 0
    iterations: int = 0
    converged: bool = False
    rms: float = float("nan")
    max_residual: float = float("nan")
    mean_residual: float = float("nan")
    sigma0: Optional[float] = None
    std: dict = field(default_factory=dict)
    residuals: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    observed: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    computed: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    image_y_flipped: bool = False
    condition_number: float = float("inf")
    warnings: list[str] = field(default_factory=list)
    cost_history: list[float] = field(default_factory=list)
    initial_guess: Optional[np.ndarray] = None

    @property
    def params_deg(self) -> np.ndarray:
        """角度为度的 6 个外方位元素。"""
        return np.array(
            [self.xs, self.ys, self.zs, self.phi, self.omega, self.kappa], dtype=float
        )

    @property
    def params_rad(self) -> np.ndarray:
        """角度为弧度的 6 个外方位元素。"""
        params = self.params_deg.copy()
        params[3:] *= DEG
        return params

    @property
    def point_residuals(self) -> np.ndarray:
        """每个点的点位残差（像坐标平面内的欧氏距离）。"""
        if self.residuals.size == 0:
            return np.zeros(0)
        return np.hypot(self.residuals[:, 0], self.residuals[:, 1])

    @property
    def rms_coordinate(self) -> float:
        """坐标分量残差 RMS（与单位权中误差同量纲，按 2n 个坐标分量统计）。"""
        if self.residuals.size == 0:
            return float("nan")
        return float(math.sqrt(float(np.sum(self.residuals ** 2)) / self.residuals.size))

    @property
    def std_degrees(self) -> dict:
        """角度精度（度）子集，便于界面直接展示。"""
        return {k: v for k, v in self.std.items() if k in ("phi", "omega", "kappa")}


def _validate_inputs(ground, image, f, x0, y0):
    ground_arr = np.asarray(ground, dtype=float)
    image_arr = np.asarray(image, dtype=float)
    if ground_arr.ndim != 2 or ground_arr.shape[1] != 3:
        raise SolveError("地面控制点必须是 n×3 的数组（每行 X Y Z）")
    if image_arr.ndim != 2 or image_arr.shape[1] != 2:
        raise SolveError("像点坐标必须是 n×2 的数组（每行 x y）")
    n = ground_arr.shape[0]
    if n != image_arr.shape[0]:
        raise SolveError(
            f"地面控制点（{n} 个）与像点（{image_arr.shape[0]} 个）数量不一致"
        )
    if n < MIN_POINTS:
        raise SolveError(f"至少需要 {MIN_POINTS} 个控制点才能解算 6 个外方位元素")
    if not np.all(np.isfinite(ground_arr)):
        raise SolveError("地面控制点中存在非数值（NaN/Inf）")
    if not np.all(np.isfinite(image_arr)):
        raise SolveError("像点坐标中存在非数值（NaN/Inf）")
    if not np.isfinite(f) or f <= 0:
        raise SolveError("相机主距必须为正数")
    if not np.isfinite(x0) or not np.isfinite(y0):
        raise SolveError("主点偏移必须为数值")

    centered = ground_arr - ground_arr.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 0 or singular[1] < 1e-6 * singular[0]:
        raise SolveError("地面控制点近似共线，无法解算外方位元素")

    # 像点重合同样会导致法方程奇异
    image_centered = image_arr - image_arr.mean(axis=0)
    image_singular = np.linalg.svd(image_centered, compute_uv=False)
    if image_singular[0] <= 0 or image_singular[1] < 1e-9 * image_singular[0]:
        raise SolveError("像点近似重合，无法解算外方位元素")

    return ground_arr, image_arr


def _affine_fit(xy: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最小二乘拟合 ``uv ≈ A·xy + t``，返回 (A, t)。"""
    design = np.column_stack((xy, np.ones(len(xy))))
    coeff, *_ = np.linalg.lstsq(design, uv, rcond=None)
    return coeff[:2].T, coeff[2]


def _detect_y_down(ground: np.ndarray, image: np.ndarray, x0: float, y0: float) -> bool:
    """通过二维对应关系的行列式符号判断像点 y 轴是否向下。

    正常的像平面坐标（x 向右、y 向上）与地面 XY 之间是相似变换，行列式为正；
    若使用像素坐标（y 向下）或 y 轴定义相反，则对应关系为镜像，行列式为负。
    """
    matrix, _ = _affine_fit(ground[:, :2], image - np.array([x0, y0]))
    return bool(np.linalg.det(matrix) < 0)


def _initial_candidates(
    ground: np.ndarray,
    image: np.ndarray,
    f: float,
    x0: float,
    y0: float,
) -> list[np.ndarray]:
    """根据二维相似关系生成多组初值（角度为弧度）。"""
    ground_xy = ground[:, :2]
    ground_center = ground_xy.mean(axis=0)
    image_centered = image - np.array([x0, y0])
    matrix, offset = _affine_fit(ground_xy, image_centered)

    scale = math.sqrt(abs(float(np.linalg.det(matrix))))
    if not np.isfinite(scale) or scale <= 1e-12:
        # 退化时用范围比值兜底
        ground_extent = float(np.sqrt(np.sum((ground_xy - ground_center) ** 2, axis=1)).mean())
        image_extent = float(
            np.sqrt(np.sum((image_centered - image_centered.mean(axis=0)) ** 2, axis=1)).mean()
        )
        scale = image_extent / ground_extent if ground_extent > 0 else 1e-3

    height = f / scale
    if not np.isfinite(height) or height <= 0:
        height = max(float(np.ptp(ground[:, 2])), 1.0) * 10.0

    try:
        camera_xy = -np.linalg.solve(matrix, offset)
    except np.linalg.LinAlgError:
        camera_xy = ground_center
    if not np.all(np.isfinite(camera_xy)):
        camera_xy = ground_center

    kappa = math.atan2(float(matrix[0, 1]), float(matrix[0, 0]))
    zs = float(ground[:, 2].mean()) + height

    base = [float(camera_xy[0]), float(camera_xy[1]), zs, 0.0, 0.0, kappa]
    candidates = [
        base,
        [base[0], base[1], base[2], 0.0, 0.0, kappa + math.pi],
        [float(ground_center[0]), float(ground_center[1]), base[2], 0.0, 0.0, kappa],
        [float(ground_center[0]), float(ground_center[1]), base[2], 0.0, 0.0, kappa + math.pi],
    ]
    # 对倾斜摄影增加几组小角度初值，降低落入局部极小的概率
    for tilt_phi, tilt_omega in ((5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)):
        candidates.append(
            [
                base[0],
                base[1],
                base[2],
                tilt_phi * DEG,
                tilt_omega * DEG,
                kappa,
            ]
        )
    return candidates


def _linearize(ground, image, params, f, x0, y0):
    design = design_matrix(ground, params, f, x0, y0)
    computed = project(ground, params, f, x0, y0)
    if not np.all(np.isfinite(computed)):
        raise SolveError("投影计算出现非数值，请检查数据是否合理")
    delta_l = (image - computed).ravel()
    return design, delta_l, computed


def _levenberg_marquardt(
    ground: np.ndarray,
    image: np.ndarray,
    initial: Sequence[float],
    f: float,
    x0: float,
    y0: float,
    max_iter: int,
    tol: float,
) -> dict:
    """从给定初值出发执行 LM 迭代，返回参数、代价、迭代次数等。"""
    params = np.asarray(initial, dtype=float).copy()
    design, delta_l, computed = _linearize(ground, image, params, f, x0, y0)
    cost = float(delta_l @ delta_l)
    cost_history = [cost]
    damping = 1e-3
    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        normal = design.T @ design
        gradient = design.T @ delta_l
        diagonal = np.maximum(np.diag(normal), 1e-12)

        accepted = False
        step = None
        for _ in range(40):
            try:
                step = np.linalg.solve(normal + damping * np.diag(diagonal), gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(
                    normal + damping * np.diag(diagonal), gradient, rcond=None
                )[0]
            if not np.all(np.isfinite(step)):
                damping *= 10.0
                continue

            trial = params + step
            try:
                trial_design, trial_l, trial_computed = _linearize(
                    ground, image, trial, f, x0, y0
                )
            except (SolveError, ValueError):
                damping *= 10.0
                continue
            trial_cost = float(trial_l @ trial_l)
            if np.isfinite(trial_cost) and trial_cost < cost:
                params = trial
                design, delta_l, computed = trial_design, trial_l, trial_computed
                improvement = cost - trial_cost
                cost = trial_cost
                damping = max(damping / 10.0, 1e-12)
                accepted = True
                break
            damping *= 10.0

        if not accepted:
            break

        cost_history.append(cost)
        if np.max(np.abs(step)) < tol:
            converged = True
            break
        if improvement <= 1e-15 * max(cost, 1.0):
            converged = True
            break
        if damping > 1e14:
            break

    return {
        "params": params,
        "cost": cost,
        "iterations": iterations,
        "converged": converged,
        "computed": computed,
        "history": cost_history,
    }


def _plane_coplanarity(ground: np.ndarray) -> float:
    """返回最小/最大奇异值之比，用于衡量控制点是否近似共面。"""
    centered = ground - ground.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 0:
        return 0.0
    return float(singular[-1] / singular[0])


def _precision(design: np.ndarray, cost: float, n: int) -> tuple[Optional[float], dict, float]:
    """计算单位权中误差、参数精度与法方程条件数。"""
    dof = 2 * n - 6
    sigma0 = math.sqrt(cost / dof) if dof > 0 and cost >= 0 else None

    normal = design.T @ design
    diag = np.diag(normal)
    if np.any(diag <= 0):
        return sigma0, {}, float("inf")

    # 归一化后求条件数，避免量纲（米/弧度）造成的误判
    inv_sqrt = 1.0 / np.sqrt(diag)
    scaled = normal * np.outer(inv_sqrt, inv_sqrt)
    try:
        condition = float(np.linalg.cond(scaled))
    except np.linalg.LinAlgError:
        condition = float("inf")

    std = {}
    if sigma0 is not None:
        try:
            covariance = np.linalg.pinv(normal) * (sigma0 ** 2)
            variances = np.diag(covariance)
            for index, name in enumerate(PARAM_NAMES):
                variance = float(variances[index])
                if variance < 0 or not np.isfinite(variance):
                    continue
                value = math.sqrt(variance)
                std[name] = value * (DEG if index >= 3 else 1.0)
        except np.linalg.LinAlgError:
            std = {}
    return sigma0, std, condition


def solve_exterior_orientation(
    ground: Sequence[Sequence[float]],
    image: Sequence[Sequence[float]],
    f: float,
    x0: float = 0.0,
    y0: float = 0.0,
    image_y_down: Optional[bool] = None,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> Solution:
    """解算单张影像的 6 个外方位元素。

    Parameters
    ----------
    ground : (n, 3) array_like
        地面控制点坐标，单位通常为米。
    image : (n, 2) array_like
        与 ``ground`` 逐行对应的像点坐标，单位需与 ``f`` 一致（通常为毫米）。
    f : float
        相机主距（与像点同单位）。
    x0, y0 : float
        主点偏移（与像点同单位），默认 0。
    image_y_down : bool or None
        ``True`` 表示像点 y 轴向下（像素坐标系），解算前会取反；
        ``None``（默认）根据对应关系自动判断；``False`` 强制按 y 轴向上处理。
    max_iter : int
        LM 最大迭代次数。
    tol : float
        参数增量收敛阈值（米 / 弧度）。

    Raises
    ------
    SolveError
        数据格式、数量或几何条件不满足时。
    """
    ground_arr, image_arr = _validate_inputs(ground, image, f, x0, y0)

    if image_y_down is None:
        flipped = _detect_y_down(ground_arr, image_arr, x0, y0)
    else:
        flipped = bool(image_y_down)

    observed = image_arr.copy()
    if flipped:
        observed[:, 1] *= -1.0

    candidates = _initial_candidates(ground_arr, observed, f, x0, y0)

    best = None
    for initial in candidates:
        try:
            result = _levenberg_marquardt(
                ground_arr, observed, initial, f, x0, y0, max_iter, tol
            )
        except (SolveError, ValueError, FloatingPointError):
            continue
        if best is None or result["cost"] < best["cost"]:
            best = result
            best["initial"] = np.asarray(initial, dtype=float)

    if best is None or not np.isfinite(best["cost"]):
        raise SolveError("迭代解算失败：无法收敛，请检查数据与单位是否匹配")

    params = best["params"]
    computed = project(ground_arr, params, f, x0, y0)
    residuals = observed - computed
    if not np.all(np.isfinite(residuals)):
        raise SolveError("解算结果异常（残差出现非数值）")

    sq_sum = float(np.sum(residuals ** 2))
    n = ground_arr.shape[0]
    # 点位残差 RMS：每个点贡献 vx² + vy²，按点数平均后开方
    rms = math.sqrt(sq_sum / n)
    point_res = np.hypot(residuals[:, 0], residuals[:, 1])

    design = design_matrix(ground_arr, params, f, x0, y0)
    sigma0, std, condition = _precision(design, sq_sum, n)

    warnings: list[str] = []
    if flipped:
        warnings.append(
            "检测到像点 y 轴与地面坐标系镜像，已按像素坐标系（y 轴向下）自动翻转后解算。"
        )
    if not best["converged"]:
        warnings.append(
            f"迭代 {best['iterations']} 次后停止，未满足严格收敛条件，结果仅供参考。"
        )
    if sigma0 is None:
        warnings.append("多余观测数为 0（仅 3 个控制点），无法进行精度评定。")

    coplanarity = _plane_coplanarity(ground_arr)
    if coplanarity < 0.02:
        warnings.append(
            "控制点近似共面，Zs 与主距 f 强相关；若 f 存在误差，高程解算误差会被放大，"
            "建议补充不同高程的控制点。"
        )

    image_radius = float(
        np.sqrt(np.mean(np.sum((observed - observed.mean(axis=0)) ** 2, axis=1)))
    )
    if image_radius > 0 and rms > 0.02 * image_radius:
        warnings.append(
            f"像点残差 RMS={rms:.4g}，相对像幅（{image_radius:.4g}）偏大，"
            "可能存在粗差、点位对应错误或单位不匹配。"
        )

    if condition > 1e12:
        warnings.append(
            f"法方程条件数较大（{condition:.3g}），控制点几何分布不佳，解算精度不稳定。"
        )
    if params[2] < float(ground_arr[:, 2].mean()):
        warnings.append(
            "解算得到的摄影中心高程低于控制点平均高程，请确认像点 y 轴方向与拍摄方位是否正确。"
        )
    if point_res.size and np.max(point_res) > 3.0 * max(rms, 1e-12):
        outlier = int(np.argmax(point_res)) + 1
        warnings.append(
            f"第 {outlier} 个点的点位残差（{np.max(point_res):.4g}）明显大于整体水平，"
            "建议检查该点的点位对应关系。"
        )

    return Solution(
        xs=float(params[0]),
        ys=float(params[1]),
        zs=float(params[2]),
        phi=float(params[3] / DEG),
        omega=float(params[4] / DEG),
        kappa=float(params[5] / DEG),
        f=float(f),
        x0=float(x0),
        y0=float(y0),
        n_points=n,
        iterations=int(best["iterations"]),
        converged=bool(best["converged"]),
        rms=float(rms),
        max_residual=float(np.max(point_res)),
        mean_residual=float(np.mean(point_res)),
        sigma0=sigma0,
        std=std,
        residuals=residuals,
        observed=observed,
        computed=computed,
        image_y_flipped=flipped,
        condition_number=condition,
        warnings=warnings,
        cost_history=best["history"],
        initial_guess=best["initial"],
    )
