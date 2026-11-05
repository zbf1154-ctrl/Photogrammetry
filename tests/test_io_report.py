"""数据读写与报告输出测试。"""

from __future__ import annotations

import numpy as np
import pytest

from photogrammetry.io import (
    DataFormatError,
    format_points,
    load_example_data,
    read_points,
    timestamp_name,
    write_report,
)
from photogrammetry.report import build_report, format_residual_table, format_summary
from photogrammetry.solver import solve_exterior_orientation


@pytest.fixture()
def solution():
    ground, image, _, _ = load_example_data()
    return solve_exterior_orientation(ground, image, 152.0)


def test_read_points_supports_comments_and_separators(tmp_path):
    path = tmp_path / "points.txt"
    path.write_text(
        "# 控制点\n100 200 30\n\n101,201,31  # 行尾注释\n102\t202\t32\n",
        encoding="utf-8",
    )
    data = read_points(path, 3, "控制点")
    assert data.shape == (3, 3)
    np.testing.assert_allclose(data[1], [101, 201, 31])


def test_read_points_supports_gbk_and_bom(tmp_path):
    gbk = tmp_path / "gbk.txt"
    gbk.write_bytes("# 控制点\n1 2 3\n".encode("gbk"))
    assert read_points(gbk, 3).shape == (1, 3)

    bom = tmp_path / "bom.txt"
    bom.write_text("# 控制点\n1 2 3\n", encoding="utf-8-sig")
    assert read_points(bom, 3).shape == (1, 3)


def test_read_points_column_mismatch_raises(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 2 3 4\n", encoding="utf-8")
    with pytest.raises(DataFormatError, match="应有 3 个数值"):
        read_points(path, 3)


def test_read_points_rejects_non_numeric_and_empty(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("1 2 abc\n", encoding="utf-8")
    with pytest.raises(DataFormatError, match="非数值"):
        read_points(bad, 3)

    empty = tmp_path / "empty.txt"
    empty.write_text("# 只有注释\n", encoding="utf-8")
    with pytest.raises(DataFormatError, match="为空"):
        read_points(empty, 3)


def test_read_points_missing_file_raises(tmp_path):
    with pytest.raises(DataFormatError, match="不存在"):
        read_points(tmp_path / "missing.txt", 3)


def test_load_example_data_has_four_points():
    ground, image, ground_path, image_path = load_example_data()
    assert ground.shape == (4, 3)
    assert image.shape == (4, 2)
    assert ground_path.name == "ground_control_points.txt"
    assert image_path.name == "image_points.txt"


def test_format_points_contains_header_and_rows():
    ground, _image, _, _ = load_example_data()
    text = format_points(ground, "ground")
    assert "地面控制点数据" in text
    assert "数据点数：4" in text
    assert "100.2500" in text


def test_write_report_uses_utf8_bom(tmp_path):
    path = tmp_path / "report.txt"
    write_report(path, "中文报告")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert "中文报告" in raw.decode("utf-8-sig")


def test_timestamp_name_pattern():
    name = timestamp_name()
    assert name.endswith(".txt")
    assert name.startswith("外方位元素解算报告_")


def test_summary_and_report_contain_key_sections(solution):
    summary = format_summary(solution)
    assert "外方位线元素" in summary
    assert "外方位角元素" in summary
    assert "Xs =" in summary and "κ =" in summary

    report = build_report(solution, "ground.txt", "image.txt")
    assert "摄影测量外方位元素解算报告" in report
    assert "旋转矩阵" in report
    assert "逐点残差" in report
    assert "控制点文件：ground.txt" in report
    assert "Zs（高程）" in report

    table = format_residual_table(solution)
    assert table.count("\n") == len(solution.residuals) + 1
