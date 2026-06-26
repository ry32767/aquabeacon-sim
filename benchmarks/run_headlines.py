"""run_headlines.py — 論文の中核数値を (コード, config, seed) にピン留めする再生成ハーネス。

README やスライドに載る見出し数値が「どの seed・config から出たか」を追跡し、後年・別環境でも
再生成・照合できるようにする (MATH_SPEC §15 / 監査 VAL-03)。fast かつ統計的に安定な基礎指標
(CRLB・効率・一貫性・理想ノイズ測位 RMSE) を計算し benchmarks/headline_metrics.json に書く。
config の実効値指紋を併記するので、config を編集すると指紋が変わり陳腐化が分かる。

実行: python benchmarks/run_headlines.py        # JSON を再生成
照合: pytest tests/test_headline_numbers.py      # 再計算して JSON と一致を確認 (CI)

注: シナリオ (run_sbl 等) の見出し RMSE は results/<名>/run_*.json に provenance つきで保存される
   (独立 RNG・95%CI 込み)。本ハーネスは高速・安定な基礎指標に限定し、論文の主張根拠を固定する。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import SIGMA, SIGMA_DEPTH
from src.evaluation import (crlb_rmse, crlb_position, monte_carlo_rmse,
                            monte_carlo_estimates, nees)
from src.results_io import _provenance

TRUTH = np.array([6.0, 8.0, -7.5])      # 公称子機位置 (d=12.5 m)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headline_metrics.json")


def compute_metrics():
    """ピン留めする基礎指標 dict を返す (fast・決定的/安定)。"""
    crlb_total = crlb_rmse(TRUTH, SIGMA) * 1000.0                 # 理論下界 [mm] (決定的)
    crlb_depth = crlb_rmse(TRUTH, SIGMA, with_depth=True,
                           sigma_depth=SIGMA_DEPTH) * 1000.0      # 深度融合の下界 [mm]
    mc_total = monte_carlo_rmse(TRUTH, SIGMA, n=2000, seed=0)["total"] * 1000.0
    est = monte_carlo_estimates(TRUTH, SIGMA, n=4000, seed=1)
    nees_mean = float(nees(TRUTH, est, crlb_position(TRUTH, SIGMA)).mean())
    return {
        "crlb_total_mm": round(crlb_total, 4),
        "crlb_total_with_depth_mm": round(crlb_depth, 4),
        "mc_rmse_total_mm": round(mc_total, 4),
        "efficiency_rmse_over_crlb": round(mc_total / crlb_total, 4),
        "nees_mean_dof3": round(nees_mean, 4),
    }


def main():
    prov = _provenance()
    record = {
        "description": "AquaBeacon 論文の中核数値ピン (CRLB/効率/一貫性/理想測位RMSE)",
        "truth_xyz": TRUTH.tolist(),
        "sigma": [float(s) for s in SIGMA],
        "config_fingerprint": prov["config_fingerprint"],
        "git_commit": prov["git"]["commit"],
        "metrics": compute_metrics(),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print("wrote", OUT)
    print(json.dumps(record["metrics"], indent=2))


if __name__ == "__main__":
    main()
