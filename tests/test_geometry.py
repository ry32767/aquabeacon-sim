"""
test_geometry.py — MATH_SPEC §6 のジオメトリ評価テストケース (Stage 2)。

- 恒等チェック: 既知キューブ (L=0.5m) の真点群をそのまま推定点群に入れると、
  寸法誤差 ≈ 0、体積誤差率 ≈ 0 になる。
- スケールチェック: 既知スケール係数 (1.02倍) をかけると、寸法誤差がその係数を
  正しく反映する。

geometry は truth を見ない (点群を入力に L_hat, V_hat を計算するだけ)。
真値との突き合わせは evaluation で行う。
"""
import numpy as np

from src.config import CUBE_SIDE, CUBE_CENTER
from src.truth import true_cube_pointcloud
from src.geometry import (aabb_dimensions, aabb_volume, convex_hull_volume,
                          cube_side_estimate)
from src.evaluation import (dimension_error_mm, volume_error_rate_pct,
                            pointcloud_rms_to_surface)

L_TRUE = CUBE_SIDE
V_TRUE = CUBE_SIDE**3


def test_aabb_dimensions_identity():
    """真キューブの AABB 各辺長 = L_true。"""
    cloud = true_cube_pointcloud()
    dims = aabb_dimensions(cloud)
    assert np.allclose(dims, [L_TRUE, L_TRUE, L_TRUE], atol=1e-9), dims


def test_dimension_error_identity():
    """恒等チェック: 寸法誤差 ≈ 0。"""
    cloud = true_cube_pointcloud()
    L_hat = cube_side_estimate(cloud)
    assert abs(dimension_error_mm(L_hat, L_TRUE)) < 1e-6


def test_volume_error_identity_aabb():
    """恒等チェック: AABB 体積の体積誤差率 ≈ 0。"""
    cloud = true_cube_pointcloud()
    V_hat = aabb_volume(cloud)
    assert abs(volume_error_rate_pct(V_hat, V_TRUE)) < 1e-6


def test_volume_error_identity_hull():
    """恒等チェック: 凸包体積でも体積誤差率 ≈ 0。"""
    cloud = true_cube_pointcloud()
    V_hat = convex_hull_volume(cloud)
    assert abs(volume_error_rate_pct(V_hat, V_TRUE)) < 1e-6


def test_pointcloud_rms_identity_is_zero():
    """同じ点群同士の点群距離 RMS は 0。"""
    cloud = true_cube_pointcloud()
    assert pointcloud_rms_to_surface(cloud, cloud) < 1e-12


def test_scaled_cube_reflects_scale_dimension():
    """スケール 1.02 倍 -> 寸法誤差が (0.51-0.50)=10mm を正しく反映する。"""
    scale = 1.02
    cloud = true_cube_pointcloud()
    center = np.asarray(CUBE_CENTER, float)
    scaled = (cloud - center) * scale + center
    L_hat = cube_side_estimate(scaled)
    assert np.isclose(L_hat, L_TRUE * scale, atol=1e-9)
    assert np.isclose(dimension_error_mm(L_hat, L_TRUE),
                      (L_TRUE * scale - L_TRUE) * 1000, atol=1e-6)   # = 10 mm


def test_scaled_cube_reflects_scale_volume():
    """スケール 1.02 倍 -> 体積誤差率 = (1.02^3 - 1)*100 ≈ 6.12%。"""
    scale = 1.02
    cloud = true_cube_pointcloud()
    center = np.asarray(CUBE_CENTER, float)
    scaled = (cloud - center) * scale + center
    V_hat = aabb_volume(scaled)
    assert np.isclose(volume_error_rate_pct(V_hat, V_TRUE),
                      (scale**3 - 1) * 100, atol=1e-6)
