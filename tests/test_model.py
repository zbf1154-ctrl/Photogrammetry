"""共线方程与旋转矩阵的数学一致性测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from photogrammetry.model import (
    camera_coordinates,
    design_matrix,
    project,
    residuals,
    rotation_derivatives,
    rotation_matrix,
)


def test_rotation_matrix_is_orthonormal():
    rng = np.random.default_rng(2026)
    for _ in range(50):
        angles = rng.uniform(-math.pi, math.pi, size=3)
        rotation = rotation_matrix(*angles)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_rotation_derivatives_match_finite_difference():
    angles = np.array([0.35, -0.62, 1.1])
    derivatives = rotation_derivatives(*angles)
    step = 1e-6
    for index in range(3):
        plus = angles.copy()
        minus = angles.copy()
        plus[index] += step
        minus[index] -= step
        numerical = (rotation_matrix(*plus) - rotation_matrix(*minus)) / (2 * step)
        np.testing.assert_allclose(derivatives[index], numerical, atol=1e-8)


def test_nadir_projection_matches_closed_form():
    f = 152.0
    camera = np.array([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])
    ground = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 0.0]])
    image = project(ground, camera, f)

    np.testing.assert_allclose(image[0], [f * 100.0 / 300.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(image[1], [0.0, f * 100.0 / 300.0], atol=1e-12)
    np.testing.assert_allclose(image[2], [0.0, 0.0], atol=1e-12)


def test_camera_coordinates_are_rotated_offsets():
    camera = np.array([10.0, 20.0, 30.0, 0.1, -0.2, 0.3])
    ground = np.array([[10.0, 20.0, 30.0], [11.0, 22.0, 33.0]])
    cam = camera_coordinates(ground, camera)
    # 摄影中心本身位于原点
    np.testing.assert_allclose(cam[0], [0.0, 0.0, 0.0], atol=1e-12)
    assert cam.shape == (2, 3)


def test_design_matrix_matches_finite_difference():
    rng = np.random.default_rng(42)
    ground = rng.uniform(0, 100, size=(6, 3))
    params = np.array([50.0, 40.0, 300.0, 0.05, -0.03, 0.4])
    f = 152.0
    analytic = design_matrix(ground, params, f)

    numerical = np.zeros_like(analytic)
    for index in range(6):
        step = 1e-5 if index < 3 else 1e-7
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        numerical[:, index] = (
            (project(ground, plus, f) - project(ground, minus, f)) / (2 * step)
        ).ravel()

    np.testing.assert_allclose(analytic, numerical, atol=1e-6)


def test_residuals_are_observed_minus_computed():
    ground = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    params = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    computed = project(ground, params, 152.0)
    observed = computed + 0.5
    np.testing.assert_allclose(residuals(ground, observed, params, 152.0), 0.5)


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        project(np.zeros((2, 2)), np.zeros(6), 152.0)
    with pytest.raises(ValueError):
        design_matrix(np.zeros((2, 3)), np.zeros(5), 152.0)
    with pytest.raises(ValueError):
        design_matrix(np.zeros((2, 3)), np.zeros(6), -1.0)
