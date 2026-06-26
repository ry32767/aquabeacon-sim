"""
test_headline_numbers.py — 論文の中核数値が再生成・照合可能であることを保証する (監査 VAL-03)。

benchmarks/headline_metrics.json にピン留めした基礎指標 (CRLB・効率・一貫性・理想測位 RMSE) を
**今のコードで再計算し**、JSON 値と許容内で一致することを確認する。リファクタや config 編集で
見出し数値が静かに変わったら CI が落ちる。JSON が無ければ skip (benchmarks/run_headlines.py で生成)。
"""
import json
import os

import numpy as np
import pytest

from benchmarks.run_headlines import compute_metrics

JSON = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "headline_metrics.json")

# 指標ごとの許容相対誤差。決定的指標 (CRLB) は厳しめ、モンテカルロ由来は緩め。
TOL = {
    "crlb_total_mm": 1e-6,
    "crlb_total_with_depth_mm": 1e-6,
    "mc_rmse_total_mm": 1e-6,        # seed 固定 + 既定 RNG で決定的
    "efficiency_rmse_over_crlb": 1e-4,
    "nees_mean_dof3": 1e-4,
}


@pytest.mark.skipif(not os.path.exists(JSON),
                    reason="benchmarks/headline_metrics.json 未生成 (run_headlines.py で生成)")
def test_headline_metrics_match_pinned():
    with open(JSON, encoding="utf-8") as f:
        pinned = json.load(f)["metrics"]
    current = compute_metrics()
    for k, ref in pinned.items():
        got = current[k]
        assert np.isclose(got, ref, rtol=TOL.get(k, 1e-4)), \
            f"見出し数値 {k} がピン値から変化: pinned={ref} now={got}"


def test_headline_efficiency_and_consistency_sane():
    """ピンに依らず、効率 (RMSE/CRLB≈1) と一貫性 (NEES≈3) が研究グレード範囲にある。"""
    m = compute_metrics()
    assert 0.9 <= m["efficiency_rmse_over_crlb"] <= 1.15
    assert 2.7 <= m["nees_mean_dof3"] <= 3.3
    assert m["crlb_total_with_depth_mm"] < m["crlb_total_mm"]   # 深度融合で下界が縮む
