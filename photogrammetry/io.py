"""数据文件读写与格式化。

文本格式约定
------------
* 每行一条记录，数值之间可用空格、制表符、逗号或分号分隔；
* ``#`` 之后的内容视为注释，整行注释或行尾注释均可；
* 自动识别 UTF-8（含 BOM）、UTF-16 与 GBK 编码，兼容 Windows 记事本文件。
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class DataFormatError(ValueError):
    """数据文件内容不符合格式要求。"""


_SEPARATOR = re.compile(r"[,\s;]+")


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DataFormatError(f"无法识别文件编码：{path.name}")


def read_points(path: str | Path, columns: int, label: str = "数据") -> np.ndarray:
    """读取点位文件并返回 ``(n, columns)`` 的浮点数组。

    Raises
    ------
    DataFormatError
        文件不存在、为空、数值个数不符或包含非数值内容时。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DataFormatError(f"文件不存在：{file_path}")

    try:
        text = _read_text(file_path)
    except OSError as exc:  # pragma: no cover - 依赖文件系统状态
        raise DataFormatError(f"文件读取失败：{exc}") from exc

    rows: list[list[float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        content = content.replace("，", ",").replace("；", ";")
        tokens = [token for token in _SEPARATOR.split(content) if token]
        if len(tokens) != columns:
            raise DataFormatError(
                f"{label}第 {line_number} 行应有 {columns} 个数值，实际为 {len(tokens)} 个："
                f"{line.strip()}"
            )
        try:
            rows.append([float(token) for token in tokens])
        except ValueError as exc:
            raise DataFormatError(
                f"{label}第 {line_number} 行包含非数值内容：{line.strip()}"
            ) from exc

    if not rows:
        raise DataFormatError(f"{label}文件为空（或全部为注释）：{file_path.name}")

    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise DataFormatError(f"{label}中包含 NaN 或无穷大")
    return data


def format_points(data: Sequence[Sequence[float]], kind: str) -> str:
    """把点位数组格式化为适合文本框展示的字符串。"""
    data = np.asarray(data, dtype=float)
    if kind == "ground":
        title, header = "地面控制点数据 (X, Y, Z)", "点号        X            Y            Z"
        rows = [
            f"{index + 1:>4}  {row[0]:>12.4f} {row[1]:>12.4f} {row[2]:>12.4f}"
            for index, row in enumerate(data)
        ]
    else:
        title, header = "像点坐标数据 (x, y)", "点号        x            y"
        rows = [
            f"{index + 1:>4}  {row[0]:>12.4f} {row[1]:>12.4f}"
            for index, row in enumerate(data)
        ]

    lines = [
        f"=== {title} ===",
        f"数据点数：{len(data)}",
        "-" * 46,
        header,
        "-" * 46,
        *rows,
    ]
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, text: str) -> None:
    """以 UTF-8（带 BOM）写入报告，保证 Windows 记事本正常显示中文。"""
    with open(Path(path), "w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(text)


def timestamp_name(prefix: str = "外方位元素解算报告") -> str:
    """生成带时间戳的默认文件名。"""
    return f"{prefix}_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"


def examples_dir() -> Path:
    """返回随项目附带的示例数据目录（不存在时返回当前目录）。"""
    candidate = Path(__file__).resolve().parent.parent / "examples"
    return candidate if candidate.is_dir() else Path.cwd()


def load_example_data() -> tuple[np.ndarray, np.ndarray, Path, Path]:
    """读取示例控制点与像点，便于一键演示。"""
    directory = examples_dir()
    ground_path = directory / "ground_control_points.txt"
    image_path = directory / "image_points.txt"
    ground = read_points(ground_path, 3, "示例地面控制点")
    image = read_points(image_path, 2, "示例像点坐标")
    return ground, image, ground_path, image_path


def format_warnings(warnings: Iterable[str]) -> str:
    """把提示信息格式化为缩进列表。"""
    return "\n".join(f"  • {item}" for item in warnings)
