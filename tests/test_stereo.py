"""
test_stereo.py — MATH_SPEC §6.2 (子機2カメラ・ステレオ) のテスト。

- 三角測量の恒等: ノイズ0なら 順変換→三角測量 が元の点に一致する。
- 精度の傾向: ベースライン大 / 距離小 で復元誤差が下がる (複数 seed 平均)。
- キューブ復元: ステレオ復元した点群の寸法・体積誤差が妥当な範囲。
- MBD: 三角測量 (geometry) は観測(方位)とカメラ位置のみを入力する。

ジオメトリ(三角測量)は truth を import しない (test_separation が別途強制)。
"""
import numpy as np
import pytest

from src.config import (STEREO_BASELINE, STEREO_SIGMA_CAM, STEREO_STANDOFF,
                        CUBE_SIDE, CUBE_CENTER)
from src.sensors import (stereo_camera_positions, simulate_stereo_observation)
from src.geometry import (stereo_triangulate, robust_cube_side_estimate,
                          robust_volume)
from src.truth import true_cube_pointcloud
from src.evaluation import (dimension_error_mm, volume_error_rate_pct,
                            pointcloud_rms_to_surface)

# §6.2 (C) 1: 三角測量の恒等 (ノイズフリー復元)
POINTS = [(0.0, 0.0, 2.0), (0.3, -0.2, 3.0), (-1.0, 0.5, 4.0), (2.0, 2.0, 5.0)]


@pytest.mark.parametrize("point", POINTS)
def test_triangulate_identity_noise_free(point):
    point = np.array(point, float)
    B = 0.1
    c_L = np.array([-B / 2, 0.0, 0.0])
    c_R = np.array([B / 2, 0.0, 0.0])
    # ノイズ0で順変換 -> 三角測量
    bearings = simulate_stereo_observation(point, c_L, c_R, sigma_cam=0.0, seed=0)
    got = stereo_triangulate(bearings, c_L, c_R)
    assert np.allclose(got, point, atol=1e-9), f"point={point} got={got}"


def test_camera_positions_baseline_and_standoff():
    """カメラ配置: 左右間隔=baseline、リグ中心=表面から standoff、視線とベースライン直交。"""
    point = np.array([1.0, 0.5, -7.0])
    center = np.array([0.0, 0.0, 0.0])
    B, S = 0.2, 2.0
    c_L, c_R = stereo_camera_positions(point, center, standoff=S, baseline=B)
    assert np.isclose(np.linalg.norm(c_R - c_L), B, atol=1e-9)
    rig = 0.5 * (c_L + c_R)
    view_dir = (point - center) / np.linalg.norm(point - center)
    # リグは表面点から standoff だけ外側
    assert np.isclose(np.linalg.norm(rig - point), S, atol=1e-9)
    # ベースラインは視線と直交
    assert abs((c_R - c_L) @ view_dir) < 1e-9


def _stereo_recover(point, center, standoff, baseline, sigma_cam, seed, looks=20):
    c_L, c_R = stereo_camera_positions(point, center, standoff, baseline)
    acc = np.zeros(3)
    for m in range(looks):
        brg = simulate_stereo_observation(point, c_L, c_R, sigma_cam, seed=seed + m)
        acc += stereo_triangulate(brg, c_L, c_R)
    return acc / looks


def test_longer_baseline_improves_accuracy():
    """ベースラインが長いほど復元誤差が小さい (複数 seed の平均)。"""
    point = np.array([0.0, 0.0, -3.0])
    center = np.zeros(3)
    err_small, err_large = [], []
    for seed in range(8):
        e_s = _stereo_recover(point, center, 3.0, 0.05, STEREO_SIGMA_CAM, seed * 100)
        e_l = _stereo_recover(point, center, 3.0, 0.40, STEREO_SIGMA_CAM, seed * 100)
        err_small.append(np.linalg.norm(e_s - point))
        err_large.append(np.linalg.norm(e_l - point))
    assert np.mean(err_large) < np.mean(err_small)


def test_closer_standoff_improves_accuracy():
    """距離 (standoff) が近いほど復元誤差が小さい (複数 seed の平均)。"""
    point = np.array([0.0, 0.0, -3.0])
    center = np.zeros(3)
    err_near, err_far = [], []
    for seed in range(8):
        e_n = _stereo_recover(point, center, 1.5, STEREO_BASELINE, STEREO_SIGMA_CAM, seed * 100)
        e_f = _stereo_recover(point, center, 6.0, STEREO_BASELINE, STEREO_SIGMA_CAM, seed * 100)
        err_near.append(np.linalg.norm(e_n - point))
        err_far.append(np.linalg.norm(e_f - point))
    assert np.mean(err_near) < np.mean(err_far)


def test_cube_stereo_reconstruction_reasonable():
    """既知キューブをステレオ復元 (多視点平均) した点群の寸法・体積誤差が妥当。

    既定 (sigma_cam 0.1deg, baseline 0.1m, standoff 2m) + 30視点平均で実用域に入る。
    """
    true_cloud = true_cube_pointcloud(n_per_edge=5)
    est = np.empty_like(true_cloud)
    for i, p in enumerate(true_cloud):
        est[i] = _stereo_recover(p, CUBE_CENTER, STEREO_STANDOFF, STEREO_BASELINE,
                                 STEREO_SIGMA_CAM, seed=1000 * i, looks=30)
    L_hat = robust_cube_side_estimate(est)
    de = abs(dimension_error_mm(L_hat, CUBE_SIDE))
    ve = abs(volume_error_rate_pct(robust_volume(est), CUBE_SIDE**3))
    rms = pointcloud_rms_to_surface(est, true_cloud) * 1000
    assert de < 50, f"dim err {de:.1f} mm"
    assert ve < 30, f"vol err {ve:.1f} %"
    assert rms < 25, f"cloud RMS {rms:.1f} mm"
