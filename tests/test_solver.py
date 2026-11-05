"""解算器行为测试：合成数据恢复、样例数据、异常与告警。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from photogrammetry.io import load_example_data
from photogrammetry.model import project
from photogrammetry.solver import SolveError, solve_exterior_orientation


def make_synthetic(tilt_deg=(0.0, 0.0, 0.0), n=8, noise=0.0, seed=7):
    """生成已知外方位元素的合成数据。"""
    rng = np.random.default_rng(seed)
    ground = rng.uniform([0.0, 0.0, 0.0], [200.0, 160.0, 25.0], size=(n, 3))
    params = np.concatenate(([110.0, 90.0, 360.0], np.radians(tilt_deg)))
    image = project(ground, params, 152.0)
    if noise:
        image = image + rng.normal(0.0, noise, size=image.shape)
    return ground, image, params


@pytest.mark.parametrize("tilt", [(0.0, 0.0, 0.0), (4.0, -3.0, 20.0), (-9.0, 6.0, -35.0)])
def test_recovers_known_orientation(tilt):
    ground, image, params = make_synthetic(tilt)
    solution = solve_exterior_orientation(ground, image, 152.0, image_y_down=False)

    assert solution.converged
    assert solution.rms < 1e-6
    np.testing.assert_allclose(solution.params_deg, np.concatenate((params[:3], np.degrees(params[3:]))), atol=1e-4)


def test_noisy_data_recovers_within_tolerance():
    ground, image, params = make_synthetic((5.0, -2.0, 30.0), noise=0.01, seed=11)
    solution = solve_exterior_orientation(ground, image, 152.0)

    expected = np.concatenate((params[:3], np.degrees(params[3:])))
    assert solution.converged
    assert solution.rms < 0.02
    assert np.allclose(solution.params_deg, expected, atol=(0.5, 0.5, 0.5, 0.05, 0.05, 0.05))


def test_sample_data_is_detected_as_y_down_and_sane():
    ground, image, _, _ = load_example_data()
    solution = solve_exterior_orientation(ground, image, 152.0, image_y_down=None)

    assert solution.image_y_flipped is True
    assert solution.converged
    assert solution.n_points == 4
    assert 300.0 < solution.zs < 400.0
    assert abs(solution.phi) < 1.0
    assert abs(solution.omega) < 1.0
    assert abs(solution.kappa) < 1.0
    assert solution.rms < 0.1
    assert any("y 轴" in item for item in solution.warnings)


def test_auto_detection_matches_explicit_choice():
    ground, image, _, _ = load_example_data()
    auto = solve_exterior_orientation(ground, image, 152.0, image_y_down=None)
    explicit = solve_exterior_orientation(ground, image, 152.0, image_y_down=True)
    np.testing.assert_allclose(auto.params_deg, explicit.params_deg, atol=1e-9)
    np.testing.assert_allclose(auto.residuals, explicit.residuals, atol=1e-12)


def test_camera_below_terrain_is_reported():
    # 强制按 y 轴向上解释镜像数据时，最优解会落在控制点下方（等效镜像），应给出提示
    ground, image, _, _ = load_example_data()
    solution = solve_exterior_orientation(ground, image, 152.0, image_y_down=False)
    assert solution.zs < ground[:, 2].mean()
    assert any("低于控制点平均高程" in item for item in solution.warnings)


def test_three_points_have_no_redundancy():
    ground, image, _ = make_synthetic(n=3, seed=3)
    solution = solve_exterior_orientation(ground, image, 152.0, image_y_down=False)
    assert solution.sigma0 is None
    assert any("多余观测" in item for item in solution.warnings)


def test_coplanar_control_points_warn():
    ground, image, _ = make_synthetic(n=6, seed=5)
    ground[:, 2] = 50.0
    image = project(ground, np.array([110.0, 90.0, 360.0, 0.0, 0.0, 0.0]), 152.0)
    solution = solve_exterior_orientation(ground, image, 152.0, image_y_down=False)
    assert any("共面" in item for item in solution.warnings)


def test_point_count_mismatch_raises():
    ground, image, _ = make_synthetic(n=5)
    with pytest.raises(SolveError, match="数量不一致"):
        solve_exterior_orientation(ground[:4], image, 152.0)


def test_too_few_points_raises():
    ground, image, _ = make_synthetic(n=2)
    with pytest.raises(SolveError, match="至少需要"):
        solve_exterior_orientation(ground, image, 152.0)


def test_collinear_points_raise():
    ground = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    image = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    with pytest.raises(SolveError, match="共线"):
        solve_exterior_orientation(ground, image, 152.0)


@pytest.mark.parametrize("f", [0.0, -152.0, float("nan")])
def test_invalid_focal_length_raises(f):
    ground, image, _ = make_synthetic()
    with pytest.raises(SolveError, match="主距"):
        solve_exterior_orientation(ground, image, f)


def test_non_finite_input_raises():
    ground, image, _ = make_synthetic()
    ground[0, 0] = float("nan")
    with pytest.raises(SolveError, match="非数值"):
        solve_exterior_orientation(ground, image, 152.0)


def test_solution_helpers_are_consistent():
    ground, image, _ = make_synthetic(noise=0.005)
    solution = solve_exterior_orientation(ground, image, 152.0)
    assert solution.params_rad.shape == (6,)
    np.testing.assert_allclose(np.degrees(solution.params_rad[3:]), solution.params_deg[3:])
    np.testing.assert_allclose(
        solution.point_residuals,
        np.hypot(solution.residuals[:, 0], solution.residuals[:, 1]),
    )
    assert math.isclose(solution.max_residual, float(solution.point_residuals.max()))
