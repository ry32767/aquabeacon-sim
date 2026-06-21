"""run_sbl.py — SBL 音響測位の比較シナリオ (MATH_SPEC §13)。

親機に音響トランスデューサを4台 (既知配置, 一辺 baseline の正方形) 搭載し、各々が子機までの
距離を測る SBL (Short BaseLine)。4点への距離 → 多辺測量で**光学の方位なしに3D測位**できる。
IMU と深度も併用する。光学追跡 (親機1カメラ+音響) との比較用シナリオ。

要点:
  - SBL は単時刻でも4距離で可観測 (単一距離フォールバック §11 の方位不定が無い)。
  - 光を使わないので水の濁り・深さの光学劣化に不感 (深い/濁った水で光学より有利になりうる)。
  - 同一平面アレイは深い子機で z が弱い → 深度センサが z を締める。

出力: results/sbl/ (図 + JSON/CSV + 自動生成 README.md)
実行: python scripts/run_sbl.py
MBD: 推定には truth を渡さず観測のみ入力。評価でだけ真値と突き合わせる。
"""
import os
import sys

import numpy as np
from _plotstyle import plt, USE_JP, JP_FONT, Lbl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_IMU, SIGMA_DEPTH, P_PARENT, SEED,
                        SBL_ANCHORS, SBL_SIGMA_RANGE, SBL_BASELINE)
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_sbl_range_sequence, simulate_observation_sequence,
                         simulate_imu_displacements, simulate_depth_sequence)
from src.estimator import (estimate_trajectory_sbl, estimate_trajectory,
                           estimate_trajectory_acoustic_inertial)
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_csv, scenario_dir, write_report

FIGDIR = scenario_dir("sbl")

DEPTHS = [5, 8, 11, 14, 17, 20]
BASELINES = [1.0, 2.0, 4.0, 6.0, 8.0]
N_SEEDS = 5
DEMO_DEPTH = 10.0           # 3D デモ (a) と ベースライン掃引 (c) の固定水深


def _traj(depth):
    return double_lawnmower_trajectory(area=(6.0, 4.0), depth=-float(depth),
                                       n_legs=2, pts_per_leg=6, origin=(3.0, 3.0))


def _anchors(baseline):
    b = baseline / 2.0
    return np.array([[b, b, 0.0], [b, -b, 0.0], [-b, b, 0.0], [-b, -b, 0.0]])


def _sbl_rmse(depth, anchors, seed):
    traj = _traj(depth)
    rng = simulate_sbl_range_sequence(traj, anchors, SBL_SIGMA_RANGE, seed=seed)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 100)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=seed + 200)
    est = estimate_trajectory_sbl(rng, anchors, SBL_SIGMA_RANGE, imu, SIGMA_IMU,
                                  dep, SIGMA_DEPTH, p_parent=P_PARENT)
    return rmse_xyz(traj, est), est, traj


def _optical_rmse(depth, seed):
    traj = _traj(depth)
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 100)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=seed + 200)
    est = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                              p_parent=P_PARENT, z_depth_seq=dep, sigma_depth=SIGMA_DEPTH)
    return rmse_xyz(traj, est)["total"] * 1000


def _fallback_rmse(depth, seed):
    traj = _traj(depth)
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 100)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=seed + 200)
    est = estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu, SIGMA_IMU,
                                                dep, SIGMA_DEPTH, p_parent=P_PARENT)
    return rmse_xyz(traj, est)["total"] * 1000


def _avg(fn):
    return float(np.mean([fn(SEED + s) for s in range(N_SEEDS)]))


def main():
    print("=== SBL 音響測位 比較シナリオ (MATH_SPEC §13) ===")
    print(f"フォント: {JP_FONT if USE_JP else '(英語ラベル)'} / アンカー4点 一辺{SBL_BASELINE:.0f}m "
          f"/ 測距σ{SBL_SIGMA_RANGE*100:.0f}cm / 平均{N_SEEDS}")

    # (b) RMSE vs 水深: SBL / optical / single-range fallback
    sbl_d, opt_d, fb_d = [], [], []
    for d in DEPTHS:
        sbl_d.append(_avg(lambda s, dd=d: _sbl_rmse(dd, SBL_ANCHORS, s)[0]["total"] * 1000))
        opt_d.append(_avg(lambda s, dd=d: _optical_rmse(dd, s)))
        fb_d.append(_avg(lambda s, dd=d: _fallback_rmse(dd, s)))
    print("\n--- RMSE total vs 水深 [mm] ---")
    print("  水深[m]                      " + "".join("%7d" % d for d in DEPTHS))
    print("  SBL  音響4点距離+IMU+深       " + "".join("%7.0f" % v for v in sbl_d))
    print("  光学 角度+音響1点距離+IMU+深  " + "".join("%7.0f" % v for v in opt_d))
    print("  単独 音響1点距離+IMU+深       " + "".join("%7.0f" % v for v in fb_d))

    # (c) RMSE vs ベースライン B (固定深)
    sbl_b = [_avg(lambda s, B=B: _sbl_rmse(DEMO_DEPTH, _anchors(B), s)[0]["total"] * 1000)
             for B in BASELINES]
    print(f"\n--- SBL RMSE vs アレイ一辺 (深さ{DEMO_DEPTH:.0f}m) [mm] ---")
    print("  一辺[m] " + "".join("%7.1f" % B for B in BASELINES))
    print("  RMSE    " + "".join("%7.0f" % v for v in sbl_b))

    # (a) 3D デモ
    r_demo, est_demo, traj_demo = _sbl_rmse(DEMO_DEPTH, SBL_ANCHORS, SEED)
    fb_demo = _avg(lambda s: _fallback_rmse(DEMO_DEPTH, s))   # 同深さの単一距離 (比較用)
    print(f"\n--- 3D軌道デモ 深さ{DEMO_DEPTH:.0f}m ---")
    print(f"  SBL RMSE total = {r_demo['total']*1000:.0f} mm "
          f"(x{r_demo['x']*1000:.0f}/y{r_demo['y']*1000:.0f}/z{r_demo['z']*1000:.0f})")

    # ===== 図 =====
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(SBL_ANCHORS[:, 0], SBL_ANCHORS[:, 1], SBL_ANCHORS[:, 2], c="k", s=70,
               marker="s", label=Lbl("親機4トランスデューサ", "4 transducers"))
    ax.plot(traj_demo[:, 0], traj_demo[:, 1], traj_demo[:, 2], "-", color="red", lw=1.8,
            label=Lbl("真の軌道", "true"))
    ax.scatter(est_demo[:, 0], est_demo[:, 1], est_demo[:, 2], c="tab:blue", s=22,
               label=Lbl("SBL推定", "SBL"))
    ax.set_title(Lbl("(a) SBL測位 深さ%.0fm RMSE %.0fmm" % (DEMO_DEPTH, r_demo["total"] * 1000),
                     "(a) SBL @ %.0fm" % DEMO_DEPTH), fontsize=10)
    ax.set_xlabel("X[m]"); ax.set_ylabel("Y[m]"); ax.set_zlabel("Z[m]")
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=30, azim=-65)

    axb = fig.add_subplot(1, 3, 2)
    axb.plot(DEPTHS, sbl_d, "o-", color="tab:blue",
             label=Lbl("SBL (音響4点距離+IMU+深)", "SBL (4 ranges)"))
    axb.plot(DEPTHS, opt_d, "s--", color="tab:green",
             label=Lbl("光学 (角度+音響1点距離+IMU+深)", "optical (angle+range)"))
    axb.plot(DEPTHS, fb_d, "^:", color="tab:orange",
             label=Lbl("単独 (音響1点距離+IMU+深)", "single range"))
    axb.set_xlabel(Lbl("水深 [m]", "depth [m]")); axb.set_ylabel("RMSE total [mm]")
    axb.set_title(Lbl("(b) RMSE vs 水深", "(b) RMSE vs depth"))
    axb.grid(alpha=0.3); axb.legend(fontsize=8)

    axc = fig.add_subplot(1, 3, 3)
    axc.plot(BASELINES, sbl_b, "o-", color="tab:blue")
    axc.set_xlabel(Lbl("アレイ一辺 baseline [m]", "array baseline [m]"))
    axc.set_ylabel("RMSE total [mm]")
    axc.set_title(Lbl("(c) SBL RMSE vs アレイ一辺 (深%.0fm)" % DEMO_DEPTH,
                      "(c) RMSE vs baseline"))
    axc.grid(alpha=0.3)

    fig.suptitle(Lbl(
        "SBL 音響測位 (親機4トランスデューサの多辺測量): 光学なしに3D測位、濁り非依存、"
        "アレイ一辺が広いほど高精度 (MATH_SPEC §13)",
        "SBL acoustic positioning (4-transducer multilateration)"))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "sbl.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "anchors_baseline_m": SBL_BASELINE, "sigma_range_m": SBL_SIGMA_RANGE,
        "rmse_vs_depth_mm": {"depths_m": DEPTHS, "sbl": sbl_d, "optical": opt_d,
                             "single_range_fallback": fb_d},
        "rmse_vs_baseline_mm": {"baselines_m": BASELINES, "sbl": sbl_b},
        "demo": {"depth_m": DEMO_DEPTH,
                 "rmse_mm": {k: r_demo[k] * 1000 for k in ("x", "y", "z", "total")}},
    }
    write_json("sbl/run_sbl", payload,
               meta={"seed": int(SEED), "n_seeds": N_SEEDS, "script": "run_sbl.py"})
    write_csv("sbl/run_sbl",
              [{"depth_m": d, "sbl_mm": round(s, 1), "optical_mm": round(o, 1),
                "single_range_mm": round(f, 1)}
               for d, s, o, f in zip(DEPTHS, sbl_d, opt_d, fb_d)],
              header=["depth_m", "sbl_mm", "optical_mm", "single_range_mm"])
    write_report(
        "sbl", "SBL 音響測位 (親機4トランスデューサ) 比較シナリオ",
        "親機に音響トランスデューサを4台 (既知配置, 一辺 baseline の正方形) 搭載し、各々が子機まで\n"
        "の距離を測る SBL。4点への距離 → 多辺測量で光学の方位なしに3D測位できる。IMU と深度も併用。\n"
        "光学追跡 (親機1カメラ+音響) および単一距離フォールバック (§11) と比較する。光を使わないので\n"
        "濁り・深さの光学劣化に不感で、4距離で水平が直接定まるぶん単一距離より高精度になりやすい。",
        condition_sections=["sbl", "noise", "depth", "trajectory"],
        outputs=[("sbl.png", "アンカー配置+軌道 / RMSE vs水深 / RMSE vs アレイ一辺"),
                 ("run_sbl.json", "全結果"),
                 ("run_sbl.csv", "水深別 RMSE (SBL/光学/単一距離)")],
        results={f"SBL 深さ{DEMO_DEPTH:.0f}m RMSE": f"{r_demo['total']*1000:.0f} mm "
                 f"(z {r_demo['z']*1000:.0f} mm)",
                 f"対 単一距離 (深さ{DEMO_DEPTH:.0f}m)":
                 f"SBL {sbl_b[BASELINES.index(SBL_BASELINE)]:.0f} mm vs 単一 {fb_demo:.0f} mm"
                 if SBL_BASELINE in BASELINES else f"SBL {r_demo['total']*1000:.0f} mm"},
        meta={"seed": SEED, "baseline_m": SBL_BASELINE}, math_spec="§13")
    print(f"\n出力 : {FIGDIR}")
    print("\n完了。SBL(4点多辺測量)が光学なしに3D測位でき、単一距離より高精度なことを確認。")


if __name__ == "__main__":
    main()
