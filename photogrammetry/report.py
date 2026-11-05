"""解算结果与报告文本的格式化。"""

from __future__ import annotations

import datetime
import math
import unicodedata
from typing import Optional, Sequence

from .model import rotation_matrix
from .solver import Solution


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _pad(text: str, width: int, align: str = "left") -> str:
    padding = max(0, width - _display_width(text))
    if align == "right":
        return " " * padding + text
    if align == "center":
        left = padding // 2
        return " " * left + text + " " * (padding - left)
    return text + " " * padding


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], aligns: Sequence[str]) -> str:
    widths = [
        max(_display_width(headers[column]), *(_display_width(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    header_line = "  ".join(
        _pad(headers[column], widths[column], aligns[column]) for column in range(len(headers))
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(_pad(row[column], widths[column], aligns[column]) for column in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _format_number(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "N/A"
    if abs(value) >= 1e6 or (0 < abs(value) < 1e-4):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _orientation_rows(solution: Solution) -> list[tuple[str, str, str, str]]:
    return [
        ("Xs（东向）", f"{_format_number(solution.xs, 4)}", "m", "摄影中心 X 坐标"),
        ("Ys（北向）", f"{_format_number(solution.ys, 4)}", "m", "摄影中心 Y 坐标"),
        ("Zs（高程）", f"{_format_number(solution.zs, 4)}", "m", "摄影中心 Z 坐标"),
        ("φ（航向角）", f"{solution.phi:.6f}", "°", "绕 Y 轴旋转"),
        ("ω（旁向角）", f"{solution.omega:.6f}", "°", "绕 X 轴旋转"),
        ("κ（像片旋角）", f"{solution.kappa:.6f}", "°", "绕 Z 轴旋转"),
    ]


def format_summary(solution: Solution) -> str:
    """生成界面展示用的结果摘要。"""
    status = "已收敛" if solution.converged else "未严格收敛"
    lines = [
        "=" * 68,
        _pad("摄影测量外方位元素解算结果", 68, "center"),
        "=" * 68,
        "",
        "【解算概况】",
        f"  相机主距 f      ：{solution.f:.4f}（与像点同单位）",
        f"  主点偏移 x0/y0  ：{solution.x0:.4f} / {solution.y0:.4f}",
        f"  控制点数量      ：{solution.n_points}",
        f"  迭代次数        ：{solution.iterations}（{status}）",
        f"  像点 y 轴方向   ：{'向下（像素坐标系，已翻转）' if solution.image_y_flipped else '向上（摄影测量坐标系）'}",
        "",
        "【外方位线元素（摄影中心坐标）】",
        f"  Xs = {_format_number(solution.xs)} m",
        f"  Ys = {_format_number(solution.ys)} m",
        f"  Zs = {_format_number(solution.zs)} m",
        "",
        "【外方位角元素（旋转角度）】",
        f"  φ = {solution.phi: .6f} °",
        f"  ω = {solution.omega: .6f} °",
        f"  κ = {solution.kappa: .6f} °",
        "",
        "【精度指标】",
        f"  点位残差 RMS    ：{_format_number(solution.rms)}（与像点同单位）",
        f"  坐标分量 RMS    ：{_format_number(solution.rms_coordinate)}",
        f"  最大点位残差    ：{_format_number(solution.max_residual)}",
    ]
    if solution.sigma0 is not None:
        lines.append(f"  单位权中误差 σ0 ：{_format_number(solution.sigma0)}")
    if solution.std:
        lines.append(
            "  参数精度（1σ）  ："
            f"Xs ±{_format_number(solution.std.get('Xs', float('nan')))} m，"
            f"Ys ±{_format_number(solution.std.get('Ys', float('nan')))} m，"
            f"Zs ±{_format_number(solution.std.get('Zs', float('nan')))} m"
        )
        lines.append(
            f"                    φ ±{_format_number(solution.std.get('phi', float('nan')), 6)}°，"
            f"ω ±{_format_number(solution.std.get('omega', float('nan')), 6)}°，"
            f"κ ±{_format_number(solution.std.get('kappa', float('nan')), 6)}°"
        )
    lines.extend(
        [
            f"  法方程条件数    ：{_format_number(solution.condition_number, 3)}",
            "",
            "【算法】",
            "  共线条件方程 + Levenberg–Marquardt 最小二乘迭代",
        ]
    )
    if solution.warnings:
        lines.extend(["", "【提示】", *[f"  • {item}" for item in solution.warnings]])
    lines.append("=" * 68)
    return "\n".join(lines) + "\n"


def format_residual_table(solution: Solution) -> str:
    """生成逐点残差表格。"""
    rows = []
    point_residuals = solution.point_residuals
    for index, (observed, computed, residual) in enumerate(
        zip(solution.observed, solution.computed, solution.residuals), start=1
    ):
        rows.append(
            [
                str(index),
                f"{observed[0]:.4f}",
                f"{observed[1]:.4f}",
                f"{computed[0]:.4f}",
                f"{computed[1]:.4f}",
                f"{residual[0]:+.4f}",
                f"{residual[1]:+.4f}",
                f"{point_residuals[index - 1]:.4f}",
            ]
        )
    headers = ["点号", "x 观测", "y 观测", "x 计算", "y 计算", "vx", "vy", "点位残差"]
    aligns = ["right"] * len(headers)
    return _table(headers, rows, aligns)


def build_report(
    solution: Solution,
    ground_path: Optional[str] = None,
    image_path: Optional[str] = None,
    when: Optional[datetime.datetime] = None,
) -> str:
    """生成可导出的完整解算报告。"""
    when = when or datetime.datetime.now()
    rotation = rotation_matrix(*solution.params_rad[3:])

    lines = [
        "=" * 68,
        _pad("摄影测量外方位元素解算报告", 68, "center"),
        "=" * 68,
        f"生成时间：{when:%Y-%m-%d %H:%M:%S}",
        f"控制点文件：{ground_path or '（未记录）'}",
        f"像点文件：{image_path or '（未记录）'}",
        "",
        "【输入参数】",
        f"  相机主距 f        ：{solution.f:.6f}（与像点同单位）",
        f"  主点偏移 x0 / y0  ：{solution.x0:.6f} / {solution.y0:.6f}",
        f"  控制点数量        ：{solution.n_points}",
        f"  像点 y 轴方向     ：{'向下（像素坐标系，已自动翻转）' if solution.image_y_flipped else '向上（摄影测量坐标系）'}",
        "",
        "【外方位线元素（摄影中心坐标）】",
        _table(
            ["参数", "数值", "单位", "说明"],
            _orientation_rows(solution)[:3],
            ["left", "right", "left", "left"],
        ),
        "",
        "【外方位角元素（旋转角度）】",
        _table(
            ["参数", "数值", "单位", "说明"],
            _orientation_rows(solution)[3:],
            ["left", "right", "left", "left"],
        ),
        "",
        "【解算状态与精度】",
        f"  迭代次数        ：{solution.iterations}（{'已收敛' if solution.converged else '未严格收敛'}）",
        f"  点位残差 RMS    ：{_format_number(solution.rms)}（与像点同单位）",
        f"  坐标分量 RMS    ：{_format_number(solution.rms_coordinate)}",
        f"  平均点位残差    ：{_format_number(solution.mean_residual)}",
        f"  最大点位残差    ：{_format_number(solution.max_residual)}",
    ]
    if solution.sigma0 is not None:
        lines.append(f"  单位权中误差 σ0 ：{_format_number(solution.sigma0)}")
    lines.append(f"  法方程条件数    ：{_format_number(solution.condition_number, 3)}")

    if solution.std:
        lines.extend(
            [
                "",
                "【参数精度（1σ，由残差估算）】",
                f"  Xs：± {_format_number(solution.std.get('Xs', float('nan')))} m",
                f"  Ys：± {_format_number(solution.std.get('Ys', float('nan')))} m",
                f"  Zs：± {_format_number(solution.std.get('Zs', float('nan')))} m",
                f"  φ ：± {_format_number(solution.std.get('phi', float('nan')), 6)} °",
                f"  ω ：± {_format_number(solution.std.get('omega', float('nan')), 6)} °",
                f"  κ ：± {_format_number(solution.std.get('kappa', float('nan')), 6)} °",
            ]
        )

    lines.extend(
        [
            "",
            "【旋转矩阵 R（地面辅助坐标系 → 像空间坐标系）】",
            *[
                "  [" + "  ".join(f"{value: .9f}" for value in row) + "]"
                for row in rotation
            ],
            "",
            "【逐点残差】（单位与像点一致"
            + ("；y 轴已按像素坐标系翻转" if solution.image_y_flipped else "")
            + "）",
            format_residual_table(solution),
            "",
            "【说明】",
            "  1. 线元素单位与地面控制点一致（通常为米）；角元素单位为度。",
            "  2. 残差 = 观测值 − 计算值；点位残差为 (vx, vy) 的欧氏距离。",
            "  3. 参数精度由单位权中误差与法方程逆矩阵估算，仅供参考。",
            "  4. 解算方法：共线条件方程 + Levenberg–Marquardt 最小二乘迭代。",
        ]
    )
    if solution.warnings:
        lines.extend(["", "【提示】", *[f"  • {item}" for item in solution.warnings]])
    lines.extend(["", "=" * 68])
    return "\n".join(lines) + "\n"
