"""test_spec.py — 設計スペックシート (run_spec.py) の境界探索ロジックの数値テスト。

掃引結果から「目標を横切る設計境界」を線形補間で正しく求められるかを検証する。
(MC や図の生成は重いのでここではロジックのみを対象にする。)
"""
import numpy as np

from scripts.run_spec import _interp_crossing, _requirement


def test_interp_crossing_linear():
    # ys = 10*xs, target=50 -> x=5
    xs = [0, 1, 2, 5, 10]
    ys = [0, 10, 20, 50, 100]
    assert _interp_crossing(xs, ys, 50) == 5.0
    # 区間内の補間
    assert _interp_crossing([0, 10], [0, 100], 25) == 2.5


def test_interp_crossing_none_when_no_cross():
    xs = [0, 1, 2]
    ys = [10, 20, 30]
    assert _interp_crossing(xs, ys, 100) is None   # 届かない


def test_requirement_le_increasing_metric():
    """指標が x とともに増加 (悪化) -> 'x <= x*' 要求。"""
    xs = [5, 10, 15, 20]
    ys = [40, 70, 100, 140]          # target=100 を x=15 で横切る
    r = _requirement("range", "m", xs, ys, 100.0, "le")
    assert r["achievable"] is True
    assert abs(r["boundary"] - 15.0) < 1e-9
    assert "<=" in r["requirement"]


def test_requirement_ge_decreasing_metric():
    """指標が x とともに減少 (改善) -> 'x >= x*' 要求。"""
    xs = [0.05, 0.1, 0.2, 0.4]
    ys = [120, 60, 25, 12]           # target=30 を 0.1〜0.2 の間で横切る
    r = _requirement("baseline", "m", xs, ys, 30.0, "ge")
    assert r["achievable"] is True
    assert 0.1 < r["boundary"] < 0.2
    assert ">=" in r["requirement"]


def test_requirement_all_pass():
    """全グリッドで目標達成 -> achievable=True, 境界はグリッド端。"""
    xs = [1, 2, 3]
    ys = [10, 12, 15]                # 全部 target=30 以下
    r = _requirement("standoff", "m", xs, ys, 30.0, "le")
    assert r["achievable"] is True
    assert r["boundary"] is None     # 範囲内に交点なし (全達成)


def test_requirement_unreachable():
    """全グリッドで未達 -> achievable=False。"""
    xs = [1, 2, 3]
    ys = [200, 210, 220]
    r = _requirement("sigma_cam", "deg", xs, ys, 30.0, "le")
    assert r["achievable"] is False
    assert r["boundary"] is None
