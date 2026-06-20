"""
test_roundtrip.py — 順変換 → 逆変換で元に戻ることを検証する (MATH_SPEC §2)。

任意の相対ベクトル v について
    inverse_observation(*forward_observation(v)) == v   (atol=1e-9)
ただし v=(0,0,0) と theta 未定義ケース (vx=vy=0) は除外する (MATH_SPEC §2 注記)。
"""
import numpy as np
import pytest

from src.sensors import forward_observation, inverse_observation


@pytest.mark.parametrize("v", [
    (3, 4, 0),
    (6, 8, -7.5),
    (2, -3, -4),
    (-5, 1, -2),
    (1, 0, 0),
    (-1, -1, -1),
    (10, -2, -15),
])
def test_roundtrip_recovers_vector(v):
    v = np.array(v, float)
    back = inverse_observation(*forward_observation(v))
    assert np.allclose(back, v, atol=1e-9), f"v={v} back={back}"


def test_roundtrip_random(seed=0):
    """ランダムな v でも往復で戻ること。真上・真下 (rho≈0) は除外。seed 固定で再現可能。"""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        v = rng.uniform(-20, 20, size=3)
        if np.hypot(v[0], v[1]) < 1e-6:   # theta 未定義ケースは除外
            continue
        back = inverse_observation(*forward_observation(v))
        assert np.allclose(back, v, atol=1e-9), f"v={v} back={back}"
