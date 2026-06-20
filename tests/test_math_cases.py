"""
test_math_cases.py — MATH_SPEC.md の数値テストケースを機械検証する。

このファイルは仕様の一部。各テストは docs/MATH_SPEC.md の (C) 数値テストケースに
1:1 対応する。実装は src/ 側 (sensors.py, estimator.py) を検証する。

対応範囲:
  §1 順変換 forward_observation
  §2 逆変換 inverse_observation
  §3 残差 residual / wrap_angle
(往復は test_roundtrip.py、推定器のノイズフリー一致は test_noise_free.py に分離。)
"""
import numpy as np
import pytest

from src.sensors import forward_observation, inverse_observation
from src.estimator import residual, wrap_angle


# ---- §1 順変換 ----
FORWARD_CASES = [
    ((3, 4, 0),      (5.0,  0.9272952180016122,  0.0)),
    ((0, 0, -10),    (10.0, 0.0,                 -1.5707963267948966)),
    ((1, 0, 0),      (1.0,  0.0,                  0.0)),
    ((0, 0, 0),      (0.0,  0.0,                  0.0)),
    ((6, 8, -7.5),   (12.5, 0.9272952180016122, -0.6435011087932844)),
]

@pytest.mark.parametrize("v, expected", FORWARD_CASES)
def test_forward_observation(v, expected):
    got = forward_observation(np.array(v, float))
    assert np.allclose(got, expected, atol=1e-9), f"v={v} got={got} exp={expected}"


# ---- §2 逆変換 ----
INVERSE_CASES = [
    ((5.0,  0.9272952180016122,  0.0),                (3.0, 4.0, 0.0)),
    ((10.0, 0.0,                 -1.5707963267948966), (0.0, 0.0, -10.0)),
    ((12.5, 0.9272952180016122, -0.6435011087932844),  (6.0, 8.0, -7.5)),
]

@pytest.mark.parametrize("z, expected_v", INVERSE_CASES)
def test_inverse_observation(z, expected_v):
    got = inverse_observation(*z)
    assert np.allclose(got, expected_v, atol=1e-9), f"z={z} got={got} exp={expected_v}"


# ---- §3 残差 ----
def test_residual_zero_when_exact():
    x = np.array([3, 4, 0], float)
    z = forward_observation(x)
    assert np.allclose(residual(x, z), 0.0, atol=1e-9)

def test_residual_distance_offset():
    x = np.array([3, 4, 0], float)
    z = np.array([5.1, 0.9272952180016122, 0.0])
    assert np.allclose(residual(x, z), [0.1, 0, 0], atol=1e-9)

def test_residual_wrap_branch():
    # MATH_SPEC §3 (C) 3行目: theta_hat=0, theta=pi の差は pi。wrap して pi のまま。
    x = np.array([1, 0, 0], float)
    z = np.array([1.0, 3.141592653589793, 0.0])
    r = residual(x, z)
    assert np.isclose(r[0], 0.0, atol=1e-6)
    assert np.isclose(abs(r[1]), np.pi, atol=1e-6)   # +pi/-pi いずれも可 (同一方向)
    assert np.isclose(r[2], 0.0, atol=1e-6)

def test_wrap_angle_branch():
    # pi と -pi は同一方向。wrap が境界で安定であること。
    assert np.isclose(wrap_angle(np.pi), np.pi, atol=1e-9) or \
           np.isclose(wrap_angle(np.pi), -np.pi, atol=1e-9)
    assert np.isclose(wrap_angle(3*np.pi), np.pi, atol=1e-9) or \
           np.isclose(wrap_angle(3*np.pi), -np.pi, atol=1e-9)
    assert np.isclose(wrap_angle(0.0), 0.0, atol=1e-9)
