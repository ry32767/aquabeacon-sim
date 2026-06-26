"""
run_minimum.py — Stage 1 (ミニマム) を1本通すスクリプト。

フロー (MBD):
  ① truth     : 子機の真位置を決める
  ② sensors   : 真値 + seed からノイズ付き観測 (d, theta, phi) を生成
  ③ estimator : 観測のみから位置を推定 (真値は渡さない)
  ⑤ evaluation: 真値 vs 推定で RMSE

ノイズフリーで RMSE≈0、ノイズ込みで MATH_SPEC §7 のオーダー (角度誤差×距離) に
なることを確認する。乱数は seed で再現可能。
"""
import os
import sys

import numpy as np

# リポジトリルートを import パスに追加 (`python scripts/run_minimum.py` 用)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SIGMA, P_PARENT, SEED, MC_N
from src.truth import true_child_position
from src.sensors import simulate_observation, forward_observation
from src.estimator import estimate_position
from src.evaluation import position_error, monte_carlo_rmse


# 注: 本スクリプトは Stage1 の**理想ベースライン**なので、config.toml の現実誤差
# ([error_model]/[acoustic]/[sync]) は意図的に反映しない (一定σの零平均ガウスのみ)。
# ノイズフリーで RMSE≈0 を確認する基準点であり、系統バイアス・音速ズレを重ねると
# その確認が崩れるため。現実誤差込みの測位評価は run_spec / run_deepwater を参照。
def main(seed: int = SEED, n_montecarlo: int = MC_N):
    # ① 真値 (Stage 1 は単一点)
    truth = true_child_position()
    d = np.linalg.norm(truth)

    # --- (a) ノイズフリー: RMSE≈0 を確認 ---
    z0 = forward_observation(truth)
    est0 = estimate_position(z0, SIGMA, p_parent=P_PARENT)
    err0 = position_error(truth, est0)

    # --- (b) ノイズ込み 単一試行 ---
    z = simulate_observation(truth, SIGMA, seed=seed, p_parent=P_PARENT)
    est = estimate_position(z, SIGMA, p_parent=P_PARENT)
    err = position_error(truth, est)

    # --- (c) ノイズ込み モンテカルロ RMSE ---
    rmse = monte_carlo_rmse(truth, SIGMA, n=n_montecarlo, seed=seed, p_parent=P_PARENT)

    print("=== Stage 1 ミニマム: 単時刻 位置推定 ===")
    print(f"truth [m]              = {np.round(truth, 4)}   (d = {d:.2f} m)")
    print()
    print("--- (a) ノイズフリー (基準: |err| ~= 0) ---")
    print(f"est   [m]              = {np.round(est0, 6)}")
    print(f"|err| total      [mm]  = {np.linalg.norm(err0)*1000:.4g}")
    print()
    print(f"--- (b) ノイズ込み 単一試行 (seed={seed}) ---")
    print(f"est   [m]              = {np.round(est, 4)}")
    print(f"|err| per axis   [mm]  = {np.round(np.abs(err)*1000, 1)}")
    print(f"|err| total      [mm]  = {np.linalg.norm(err)*1000:.1f}")
    print()
    print(f"--- (c) モンテカルロ RMSE (n={n_montecarlo}, seed={seed}) ---")
    print(f"RMSE x/y/z       [mm]  = "
          f"{rmse['x']*1000:.1f} / {rmse['y']*1000:.1f} / {rmse['z']*1000:.1f}")
    print(f"RMSE total       [mm]  = {rmse['total']*1000:.1f}")
    print(f"参考 d*sigma_ang [mm]  = {d*SIGMA[1]*1000:.1f}  "
          f"(横方向誤差のオーダー目安, MATH_SPEC §7)")

    return rmse


if __name__ == "__main__":
    main()
