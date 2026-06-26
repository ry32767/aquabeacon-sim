"""
run_sensitivity.py — 感度解析。実機実験の設計根拠を出す。

何を振るか:
  - ノイズ sigma_angle: 角度精度がどこまで効くか
  - 距離 d: 角度誤差は d に比例して位置誤差に効く (横誤差 ≈ d * sigma_angle)
  - 仰角 phi (真下付近 rho≈0): 精度が落ちないか

各条件で N 試行のモンテカルロを回し、RMSE を集計して表に出す。
src の sensors/estimator/evaluation を使い、seed を固定して再現可能にする。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, MC_N, TRUE_CHILD_POSITION, SENS_DEPTH_Z,
                        SENS_ANGLE_DEGS, SENS_ELEV_DEGS, SENS_NADIR_D)
from src.evaluation import monte_carlo_rmse

# 注: 本スクリプトは1パラメータ (σ・距離・仰角) を**単独で振る制御掃引**なので、
# config.toml の現実誤差 ([error_model]/[acoustic]/[sync]) は意図的に反映しない。
# 系統バイアス・外れ値・音速ズレを重ねると「横誤差 ~ d*σ_ang」の素直な関係が濁るため。
# 現実誤差込みの設計掃引は run_spec を参照。


def _rmse_mm(truth, sigma, n=MC_N, seed=0):
    return monte_carlo_rmse(truth, sigma, n=n, seed=seed)['total'] * 1000


def sweep_distance():
    print("\n== 距離 d を振る (斜め配置。横方向は angle 誤差が支配) ==")
    print(f"{'depth z [m]':>12} {'d [m]':>8} {'RMSE [mm]':>10} {'d*sig_ang [mm]':>14}")
    for z in SENS_DEPTH_Z:
        truth = np.array([abs(z) * 0.6, abs(z) * 0.3, z])   # 斜め配置
        d = np.linalg.norm(truth)
        rmse = _rmse_mm(truth, SIGMA, seed=1)
        print(f"{z:>12.1f} {d:>8.2f} {rmse:>10.1f} {d*SIGMA[1]*1000:>14.1f}")


def sweep_noise():
    truth = np.asarray(TRUE_CHILD_POSITION, float)
    print(f"\n== 角度ノイズを振る (truth={np.round(truth,1)}, "
          f"d={np.linalg.norm(truth):.1f}m 固定) ==")
    print(f"{'sigma_ang [deg]':>16} {'RMSE [mm]':>10}")
    for deg in SENS_ANGLE_DEGS:
        sigma = (SIGMA[0], np.deg2rad(deg), np.deg2rad(deg))
        print(f"{deg:>16.2f} {_rmse_mm(truth, sigma, seed=2):>10.1f}")


def sweep_nadir():
    d = SENS_NADIR_D
    print(f"\n== 仰角を振る (真下 rho~=0 付近で破綻しないか。d={d:.0f}m 固定) ==")
    print(f"{'phi [deg]':>10} {'rho [m]':>9} {'RMSE [mm]':>10}")
    for phi_deg in SENS_ELEV_DEGS:
        phi = np.deg2rad(phi_deg)
        rho = d * np.cos(phi)
        truth = np.array([rho, 0.0, d * np.sin(phi)])   # theta=0 平面
        rmse = _rmse_mm(truth, SIGMA, seed=3)
        print(f"{phi_deg:>10.1f} {rho:>9.3f} {rmse:>10.1f}")


if __name__ == "__main__":
    sweep_distance()
    sweep_noise()
    sweep_nadir()
    print("\n注: 横方向誤差 ~= d * sigma_angle のオーダーになることを確認 (MATH_SPEC §7)。")
