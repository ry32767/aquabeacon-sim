"""run_robust.py — ロバスト推定のデモ (MATH_SPEC §4.4)。

外れ値 (ライト見失い・音響マルチパス) を数時刻に注入したダブル芝刈り軌道を、
純L2 (linear) と各ロバスト損失 (huber/soft_l1/cauchy) で推定し、RMSE を比較する。
IMU 拘束つき軌道推定では、ロバスト損失が外れ値時刻の残差を減衰し RMSE を大きく下げる。

出力: コンソール表 + figures/robust/robust_vs_linear.png + results/run_robust.{json,csv}

実行: python scripts/run_robust.py
MBD: 推定 (estimate_trajectory) には truth を渡さず観測のみ入力。評価でだけ突き合わせる。
"""
import os
import sys

import numpy as np
from _plotstyle import plt, USE_JP, JP_FONT, Lbl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SIGMA, SIGMA_IMU, P_PARENT, SEED
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_observation_sequence, simulate_imu_displacements,
                         apply_attitude_error_config)
from src.estimator import estimate_trajectory
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_csv, scenario_dir, write_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = scenario_dir("robust")


# 注入する外れ値: (時刻index, 成分index 0=d/1=az/2=el, 加算量)
OUTLIERS = [
    (3, 0, 3.0),                 # 距離 +3.0 m (音響マルチパス相当)
    (3, 1, np.deg2rad(18.0)),    # 方位 +18 deg (ライト誤検出相当)
    (7, 0, -2.5),                # 距離 -2.5 m
    (7, 2, np.deg2rad(15.0)),    # 仰角 +15 deg
]
LOSSES = ["linear", "huber", "soft_l1", "cauchy"]


def build_case(seed=SEED):
    traj = double_lawnmower_trajectory(area=(6.0, 4.0), depth=-7.5,
                                       n_legs=2, pts_per_leg=5, origin=(3.0, 4.0))
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    z = apply_attitude_error_config(z, seed=seed)        # §14 波動揺 (config [attitude].as_error。既定 OFF)
    for k, comp, val in OUTLIERS:
        z[k, comp] += val
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 1)
    outlier_times = sorted({k for k, _, _ in OUTLIERS})
    return traj, z, imu, outlier_times


def main(seed=SEED):
    print("=== ロバスト推定デモ (MATH_SPEC §4.4) ===")
    print(f"フォント: {JP_FONT if USE_JP else '(英語ラベル)'}")
    traj, z, imu, outlier_times = build_case(seed)
    print(f"軌道点数 n = {len(traj)} / 外れ値を注入した時刻 = {outlier_times}")

    results = {}
    estimates = {}
    for loss in LOSSES:
        est = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss=loss)
        estimates[loss] = est
        results[loss] = rmse_xyz(traj, est)["total"] * 1000

    base = results["linear"]
    print("\n  損失        RMSE total   linear比")
    for loss in LOSSES:
        imp = (1 - results[loss] / base) * 100
        print(f"  {loss:8s}   {results[loss]:6.1f} mm   {imp:+5.1f} %")
    best = min((l for l in LOSSES if l != "linear"), key=lambda l: results[l])
    print(f"\n  -> ロバスト損失 '{best}' で RMSE {base:.0f} -> {results[best]:.0f} mm "
          f"({(1-results[best]/base)*100:.0f}% 改善)")

    # --- 図: linear vs best robust の軌道 ---
    est_lin, est_rob = estimates["linear"], estimates[best]
    fig = plt.figure(figsize=(11, 5.5))
    for j, (title, est, col) in enumerate([
            (Lbl("純L2 (linear): 外れ値に引っ張られる", "linear: pulled by outliers"),
             est_lin, "gray"),
            (Lbl("ロバスト '%s': 外れ値を減衰" % best, "robust '%s'" % best),
             est_rob, "tab:blue")]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.8,
                label=Lbl("真の軌道", "true"))
        ax.scatter(est[:, 0], est[:, 1], est[:, 2], c=col, s=24,
                   label=Lbl("推定", "estimate"))
        for t in outlier_times:        # 外れ値時刻を強調
            ax.scatter(*est[t], c="tab:orange", s=90, marker="x")
            ax.plot([traj[t, 0], est[t, 0]], [traj[t, 1], est[t, 1]],
                    [traj[t, 2], est[t, 2]], "tab:orange", lw=1.0)
        ax.set_title("%s\nRMSE %.0f mm" % (title, results["linear" if j == 0 else best]),
                     fontsize=10)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.legend(fontsize=8, loc="upper left")
        ax.view_init(elev=40, azim=-65)
    fig.suptitle(Lbl(
        "ロバスト推定: IMU拘束つき軌道で外れ値の影響を抑える (×=外れ値時刻)",
        "Robust estimation suppresses outliers (x = outlier times)"))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "robust_vs_linear.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # --- 機械可読出力 ---
    payload = {"rmse_total_mm": results, "best_robust": best,
               "outlier_times": outlier_times,
               "improvement_pct": (1 - results[best] / base) * 100}
    jpath = write_json("robust/run_robust", payload,
                       meta={"seed": int(seed), "script": "run_robust.py"})
    cpath = write_csv("robust/run_robust",
                      [{"loss": l, "rmse_total_mm": results[l]} for l in LOSSES],
                      header=["loss", "rmse_total_mm"])
    rpath = write_report(
        "robust", "ロバスト推定デモ (外れ値耐性)",
        "外れ値 (ライト見失い・音響マルチパス) を数時刻に注入したダブル芝刈り軌道を、純L2 と\n"
        "各ロバスト損失 (huber/soft_l1/cauchy) で推定し RMSE を比較する。IMU拘束つき軌道推定で\n"
        "ロバスト損失が外れ値時刻の残差を減衰し、RMSE を大きく下げることを示す。",
        condition_sections=["noise", "montecarlo", "attitude"],
        not_reflected=[
            ("`[error_model]` (バイアス/距離成長/外れ値/音速ズレ/遅延)",
             "本シナリオは**固定の制御外れ値**を所定の時刻に注入し、純L2 vs ロバスト損失の効果を"
             "切り分けて見る制御実験。config のランダム外れ値や系統誤差を重ねると比較が濁るため反映しない。"),
            ("`[estimator]` (loss/f_scale)",
             "純L2 と各ロバスト損失 (huber/soft_l1/cauchy) を**意図的に掃引比較**するのが目的なので、"
             "config の `loss` は使わない (f_scale は estimator 既定 1.345)。"),
            ("`[trajectory]`",
             "外れ値の効果を切り分けるため固定の小さな制御軌道 (6×4m, 2レグ×5点) を使う。"
             "標準のダブル芝刈り軌道 (`[trajectory]`) は `run_mapping` を参照。"),
            ("`[optical]`", "光減衰モデルは使わない (一定σ)。減衰込みのロバスト効果は `run_deepwater`。"),
            ("`[depth]`/`[sbl]`/`[attitude]`", "深度・SBL・親機姿勢は使わない (別シナリオ)。"),
        ],
        outputs=[("robust_vs_linear.png", "純L2 vs ロバストの軌道比較"),
                 ("run_robust.json", "全損失の RMSE"),
                 ("run_robust.csv", "損失別 RMSE 表")],
        results={"純L2 RMSE": f"{base:.0f} mm",
                 f"最良ロバスト ({best})": f"{results[best]:.0f} mm",
                 "改善率": f"{(1-results[best]/base)*100:.0f} %"},
        meta={"seed": seed}, math_spec="§4.4")
    print(f"\n出力 : {FIGDIR}\n  {os.path.basename(jpath)} / {os.path.basename(cpath)} / "
          f"{os.path.basename(rpath)}")
    print("\n完了。外れ値下で robust 損失が L2 より高精度なことを確認。")


if __name__ == "__main__":
    main()
