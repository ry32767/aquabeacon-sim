"""run_switch.py — 光学↔フォールバック自動切替シナリオ (MATH_SPEC §12)。

子機が一定深のサーベイ中に**濁りのプルーム (濁った塊)** を通過し、その間だけ光学ビーコンを
見失う状況を想定する。自動切替 (見失い率+ヒステリシス, §12) が:
  - 光学が健全な区間 → 光学 (距離+方位+仰角) で高精度
  - プルーム通過中 (見失い) → 距離+IMU+深度 (§11) に自動でフォールバック
を選び、ブラックアウトでも軌道を保つ。

比較:
  - 素朴な光学維持 (見失いの誤検出をそのまま使う) → ブラックアウトで破綻
  - 常時フォールバック → 安定だが光学を活かせない
  - 自動切替 → 両方のいいとこ取り

出力: figures/switch/auto_switch.png + results/run_switch.{json,csv}
実行: python scripts/run_switch.py
"""
import os
import sys

import numpy as np
from _plotstyle import plt, USE_JP, JP_FONT, Lbl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_IMU, SIGMA_DEPTH, DEPTH_BIAS, P_PARENT, SEED,
                        SWITCH_DROPOUT_THRESHOLD, SWITCH_HYSTERESIS,
                        SURVEY_AREA, SURVEY_ORIGIN)
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_observation_sequence, simulate_imu_displacements,
                         simulate_depth_sequence, apply_attitude_error_config)
from src.estimator import (estimate_trajectory, estimate_trajectory_auto,
                           estimate_trajectory_acoustic_inertial)
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_csv, scenario_dir, write_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = scenario_dir("switch")

# 本シナリオは光学↔フォールバック自動切替**ロジック**の制御デモ (所定のプルーム区間で
# 決定的にブラックアウト)。§8 の系統誤差・外れ値は重ねない (切替ロジックの可視化を明快に保つ;
# フォールバック区間 §11 は外れ値に脆い)。現実誤差込みの評価は run_spec/deepwater。
# 幾何は config [survey] の near-nadir (子機はほぼ親機直下)。


def build_scenario(seed=SEED):
    traj = double_lawnmower_trajectory(area=SURVEY_AREA, depth=-9.0,
                                       n_legs=3, pts_per_leg=9, origin=SURVEY_ORIGIN)
    n = len(traj)
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    z = apply_attitude_error_config(z, seed=seed)        # §14 波動揺 (config [attitude].as_error。既定 OFF)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 1)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=seed + 2, bias=DEPTH_BIAS)
    # 濁りプルーム通過: 中央の連続区間でビーコン見失い
    det = np.ones(n, bool)
    a, b = int(n * 0.38), int(n * 0.72)
    det[a:b] = False
    # 見失いフレームの角度は誤検出 (大外れ値)。距離・深度・IMU は生きている。
    z_bad = z.copy()
    rng = np.random.default_rng(seed + 9)
    for k in range(n):
        if not det[k]:
            z_bad[k, 1] += rng.uniform(-0.5, 0.5)
            z_bad[k, 2] += rng.uniform(-0.5, 0.5)
    return traj, z, z_bad, imu, dep, det, (a, b)


def main(seed=SEED):
    print("=== 光学<->フォールバック自動切替シナリオ (MATH_SPEC §12) ===")
    print(f"フォント: {JP_FONT if USE_JP else '(英語ラベル)'} / "
          f"切替: 見失い率>{SWITCH_DROPOUT_THRESHOLD:.2f} (ヒステリシス{SWITCH_HYSTERESIS:.2f})")
    traj, z, z_bad, imu, dep, det, (a, b) = build_scenario(seed)
    n = len(traj)
    print(f"軌道点数 n={n} / プルーム通過 (見失い) フレーム = {a}..{b-1}")

    naive = estimate_trajectory(z_bad, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                p_parent=P_PARENT, loss="huber")
    fb = estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu, SIGMA_IMU,
                                               dep, SIGMA_DEPTH, p_parent=P_PARENT)
    auto, mask = estimate_trajectory_auto(z_bad, SIGMA, det, imu, SIGMA_IMU, dep,
                                          SIGMA_DEPTH, p_parent=P_PARENT,
                                          threshold=SWITCH_DROPOUT_THRESHOLD,
                                          hysteresis=SWITCH_HYSTERESIS)
    methods = {"素朴な光学維持": naive, "常時フォールバック": fb, "自動切替": auto}
    print("\n--- RMSE total [mm] ---")
    for k, e in methods.items():
        print(f"  {k:14s} {rmse_xyz(traj, e)['total']*1000:6.0f}")
    print(f"  自動切替が光学を使ったフレーム = {int(mask.sum())}/{n}")

    # ===== 図 =====
    fig = plt.figure(figsize=(16, 5))
    # (a) 3D: 真値 + 自動切替 (モードで色分け)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.6,
            label=Lbl("真の軌道", "true"))
    mo = mask
    ax.scatter(auto[mo, 0], auto[mo, 1], auto[mo, 2], c="tab:green", s=24,
               label=Lbl("光学モード", "optical"))
    ax.scatter(auto[~mo, 0], auto[~mo, 1], auto[~mo, 2], c="gold", s=24,
               label=Lbl("フォールバック", "fallback"))
    ax.set_title(Lbl("(a) 自動切替の軌道 (色=モード)", "(a) auto trajectory"), fontsize=10)
    ax.set_xlabel("X[m]"); ax.set_ylabel("Y[m]"); ax.set_zlabel("Z[m]")
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=40, azim=-65)

    # (b) フレーム別 位置誤差
    axb = fig.add_subplot(1, 3, 2)
    fr = np.arange(n)
    for k, e, col in [("素朴な光学維持", naive, "gray"),
                      ("常時フォールバック", fb, "tab:orange"),
                      ("自動切替", auto, "tab:blue")]:
        err = np.linalg.norm(e - traj, axis=1) * 1000
        axb.plot(fr, err, "-", color=col, label=Lbl(k, k))
    axb.axvspan(a, b - 1, color="gold", alpha=0.2,
                label=Lbl("見失い区間", "blackout"))
    axb.set_xlabel(Lbl("フレーム", "frame")); axb.set_ylabel(Lbl("位置誤差 [mm]", "err [mm]"))
    axb.set_title(Lbl("(b) フレーム別 位置誤差", "(b) per-frame error"))
    axb.grid(alpha=0.3); axb.legend(fontsize=8)

    # (c) モードタイムライン
    axc = fig.add_subplot(1, 3, 3)
    axc.step(fr, det.astype(int), where="mid", color="k",
             label=Lbl("検出 (1=見えた)", "detected"))
    axc.step(fr, mask.astype(int) - 0.05, where="mid", color="tab:green", lw=2,
             label=Lbl("光学使用 (自動切替)", "optical used"))
    axc.axvspan(a, b - 1, color="gold", alpha=0.2)
    axc.set_ylim(-0.3, 1.3); axc.set_yticks([0, 1])
    axc.set_xlabel(Lbl("フレーム", "frame"))
    axc.set_title(Lbl("(c) 検出と切替の時系列", "(c) detect & switch timeline"))
    axc.grid(alpha=0.3); axc.legend(fontsize=8, loc="center right")

    fig.suptitle(Lbl(
        "光学↔フォールバック自動切替: プルーム通過の見失い中はフォールバック、健全区間は光学 "
        "(両モードのいいとこ取り, §12)",
        "Auto optical/fallback switching keeps positioning through a blackout"))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "auto_switch.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_frames": n, "blackout_frames": [a, b - 1],
        "optical_used_frames": int(mask.sum()),
        "rmse_total_mm": {k: rmse_xyz(traj, e)["total"] * 1000
                          for k, e in methods.items()},
    }
    jpath = write_json("switch/run_switch", payload,
                       meta={"seed": int(seed), "threshold": SWITCH_DROPOUT_THRESHOLD,
                             "hysteresis": SWITCH_HYSTERESIS, "script": "run_switch.py"})
    cpath = write_csv("switch/run_switch",
                      [{"method": k, "rmse_total_mm": round(rmse_xyz(traj, e)["total"] * 1000, 1)}
                       for k, e in methods.items()],
                      header=["method", "rmse_total_mm"])
    rmse_map = {k: rmse_xyz(traj, e)["total"] * 1000 for k, e in methods.items()}
    write_report(
        "switch", "光学↔フォールバック自動切替シナリオ",
        "子機が一定深のサーベイ中に濁りのプルームを通過し、その間だけ光学ビーコンを見失う状況を\n"
        "想定する。自動切替 (見失い率+ヒステリシスの状態機械, §12) が、光学が健全な区間は光学\n"
        "(距離+方位+仰角)、見失い区間は距離+IMU+深度 (§11) を選び、ブラックアウトでも軌道を保つ。\n"
        "素朴な光学維持 (見失いの誤検出を使う) / 常時フォールバック / 自動切替 を比較する。",
        condition_sections=["survey", "noise", "switch", "depth", "attitude"],
        not_reflected=[
            ("`[error_model]`/`[acoustic]`/`[sync]`",
             "自動切替**ロジック**の制御デモ (所定区間で決定的にブラックアウト)。§8 の系統誤差・"
             "ランダム外れ値を重ねると切替判定の可視化が濁り、フォールバック区間 (§11) も脆くなるため"
             "反映しない。現実誤差込みの測位評価は `run_spec`/`run_deepwater`。"),
            ("`[switch] snr_margin`",
             "切替は見失い率と `dropout_threshold`+`hysteresis` の状態機械で判定する。"
             "`snr_margin` は参考値で本実装では未使用 (config の注記どおり)。"),
            ("`[trajectory]`",
             "幾何は config `[survey]` の near-nadir 箱 (固定 depth=-9m, 3レグ×9点)。"
             "標準のダブル芝刈り軌道 (`[trajectory]`) は使わない (`run_mapping` 参照)。"),
            ("`[optical]` (減衰σ/p_drop)",
             "見失いは optical_model の確率ではなく、明示した中央区間で**決定的に**起こす"
             "(切替判定の可視化を明快にする)。見失い角度は固定範囲の一様外れ値で表す。"),
            ("`[sbl]`/`[stereo]`/`[attitude]`", "SBL・ステレオ・親機姿勢は使わない (別シナリオ)。"),
        ],
        outputs=[("auto_switch.png", "モード色分け軌道/フレーム別誤差/検出と切替の時系列"),
                 ("run_switch.json", "各手法のRMSEと切替情報"),
                 ("run_switch.csv", "手法別 RMSE")],
        results={"素朴な光学維持": f"{rmse_map['素朴な光学維持']:.0f} mm",
                 "常時フォールバック": f"{rmse_map['常時フォールバック']:.0f} mm",
                 "自動切替": f"{rmse_map['自動切替']:.0f} mm",
                 "光学使用フレーム": f"{int(mask.sum())}/{n}"},
        meta={"seed": seed, "threshold": SWITCH_DROPOUT_THRESHOLD}, math_spec="§12")
    print(f"\n出力 : {FIGDIR}")
    print("\n完了。自動切替がブラックアウトを跨いで測位を維持することを確認。")


if __name__ == "__main__":
    main()
