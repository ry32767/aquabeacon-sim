"""run_no_optical.py — 光学なしフォールバックのテストシナリオ (MATH_SPEC §11)。

光学追跡が使えない/失われた場合 (濁り水でビーコン見失い = §9 の検出律速) を想定し、
**音響距離 + IMU + 深度センサのみ**で子機軌道を推定する。

要点:
  - 単時刻は方位が不可観測 (距離+深度の2拘束)。IMU で時刻間を繋ぐと軌道が可観測。
  - この推定は**光を使わない**ので、水の濁り・深さによる光学劣化 (§9) の影響を受けない。
    → 光学が見失う検出限界より深く/濁った水でも、同じ精度で測位を続けられる。

出力: コンソール表 + figures/no_optical/no_optical.png + results/run_no_optical.{json,csv}
実行: python scripts/run_no_optical.py
MBD: 推定には truth を渡さず観測のみ入力。評価でだけ真値と突き合わせる。
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_IMU, SIGMA_DEPTH, P_PARENT, SEED,
                        OPTICAL_MODEL, DEEP_CLARITIES)
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_observation_sequence, simulate_imu_displacements,
                         simulate_depth_sequence, optical_angular_sigma, optical_snr)
from src.estimator import (estimate_trajectory, estimate_trajectory_acoustic_inertial)
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures", "no_optical")
os.makedirs(FIGDIR, exist_ok=True)
_JP_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "Hiragino Sans", "TakaoPGothic", "IPAexGothic"]
_available = {f.name for f in fm.fontManager.ttflist}
_JP = next((c for c in _JP_CANDIDATES if c in _available), None)
USE_JP = _JP is not None
if USE_JP:
    plt.rcParams["font.family"] = _JP
plt.rcParams["axes.unicode_minus"] = False

CLARITY_LABEL = {0.05: "clear", 0.3: "coastal", 1.0: "turbid"}
DEPTHS = [5, 8, 11, 14, 17, 20]      # 水深スキャン [m]
N_SEEDS = 5                          # 各条件の平均試行数
DEMO_DEPTH = 17.0                    # 3D軌道パネルの水深 (多くの濁りで光学は既に不可)


def Lbl(ja, en):
    return ja if USE_JP else en


def _traj_at_depth(depth):
    return double_lawnmower_trajectory(area=(6.0, 4.0), depth=-float(depth),
                                       n_legs=2, pts_per_leg=6, origin=(3.0, 3.0))


def _no_optical_estimate(traj, seed):
    """距離+IMU+深度のみで軌道推定 (光学なし)。"""
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 100)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=seed + 200)
    est = estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu, SIGMA_IMU,
                                                dep, SIGMA_DEPTH, p_parent=P_PARENT)
    return est


def _optical_estimate(traj, depth, clarity, seed):
    """光学あり (角度+距離+IMU+深度) で軌道推定。光学σは減衰モデル(§9)で校正。"""
    r = float(np.linalg.norm(traj.mean(axis=0)))
    m = dict(OPTICAL_MODEL); m["attenuation_c"] = clarity
    s_ang = optical_angular_sigma(r, m)
    sig = (SIGMA[0], s_ang, s_ang)
    z = simulate_observation_sequence(traj, sig, seed=seed, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 100)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=seed + 200)
    return estimate_trajectory(z, sig, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                               p_parent=P_PARENT, z_depth_seq=dep,
                               sigma_depth=SIGMA_DEPTH)


def _avg_rmse(fn):
    """N_SEEDS 平均の per-axis RMSE [mm] を返す。"""
    acc = {k: [] for k in ("x", "y", "z", "total")}
    for s in range(N_SEEDS):
        traj, est = fn(SEED + s)
        r = rmse_xyz(traj, est)
        for k in acc:
            acc[k].append(r[k] * 1000)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _detection_limit(clarity):
    """光学が見失う水深 (SNR<snr_min となる最浅水深) [m]。"""
    m = dict(OPTICAL_MODEL); m["attenuation_c"] = clarity
    for d in np.arange(3.0, 30.0, 0.5):
        truth = np.array([3.0, 2.0, -d])
        if optical_snr(float(np.linalg.norm(truth)), m) < m["snr_min"]:
            return float(d)
    return 30.0


def main():
    print("=== 光学なしフォールバック (距離+IMU+深度) シナリオ (MATH_SPEC §11) ===")
    print(f"フォント: {_JP if USE_JP else '(英語ラベル)'} / σ_depth={SIGMA_DEPTH*100:.0f}cm "
          f"/ 平均試行={N_SEEDS}")

    # --- 水深スキャン: 光学なし (濁り非依存) ---
    no_opt = {}
    for d in DEPTHS:
        traj = _traj_at_depth(d)
        no_opt[d] = _avg_rmse(lambda s, t=traj: (t, _no_optical_estimate(t, s)))
    print("\n--- 光学なし (距離+IMU+深度) RMSE [mm] / 濁りに依存しない ---")
    print("  水深[m] " + "".join("%7d" % d for d in DEPTHS))
    print("  x       " + "".join("%7.0f" % no_opt[d]["x"] for d in DEPTHS))
    print("  z       " + "".join("%7.0f" % no_opt[d]["z"] for d in DEPTHS))
    print("  total   " + "".join("%7.0f" % no_opt[d]["total"] for d in DEPTHS))

    # --- 光学あり (coastal) を比較用に: 検出限界まで ---
    det_limits = {c: _detection_limit(c) for c in DEEP_CLARITIES}
    opt_coastal = {}
    for d in DEPTHS:
        if d <= det_limits[0.3]:
            traj = _traj_at_depth(d)
            opt_coastal[d] = _avg_rmse(
                lambda s, t=traj, dd=d: (t, _optical_estimate(t, dd, 0.3, s)))["total"]
    print("\n--- 検出限界 (光学が見失う水深) ---")
    for c in DEEP_CLARITIES:
        print(f"  {CLARITY_LABEL.get(c,'c'):8s}(c={c:.2f}): {det_limits[c]:.1f} m")

    # --- 3D軌道デモ (DEMO_DEPTH) ---
    traj_demo = _traj_at_depth(DEMO_DEPTH)
    est_demo = _no_optical_estimate(traj_demo, SEED)
    rmse_demo = rmse_xyz(traj_demo, est_demo)
    print(f"\n--- 3D軌道デモ 深さ{DEMO_DEPTH:.0f}m (多くの濁りで光学は既に不可) ---")
    print(f"  光学なし RMSE total = {rmse_demo['total']*1000:.0f} mm "
          f"(x{rmse_demo['x']*1000:.0f}/y{rmse_demo['y']*1000:.0f}/z{rmse_demo['z']*1000:.0f})")

    # ===== 図 =====
    fig = plt.figure(figsize=(16, 5))
    # (a) 3D 軌道
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(traj_demo[:, 0], traj_demo[:, 1], traj_demo[:, 2], "-", color="red",
            lw=1.8, label=Lbl("真の軌道", "true"))
    ax.scatter(est_demo[:, 0], est_demo[:, 1], est_demo[:, 2], c="tab:blue", s=22,
               label=Lbl("光学なし推定", "no-optical"))
    ax.set_title(Lbl("(a) 光学なし軌道 深さ%.0fm  RMSE %.0fmm" %
                     (DEMO_DEPTH, rmse_demo["total"] * 1000),
                     "(a) no-optical @ %.0fm" % DEMO_DEPTH), fontsize=10)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=38, azim=-65)

    # (b) RMSE total vs 水深: 光学なし(平坦) + 光学(coastal) + 検出限界
    axb = fig.add_subplot(1, 3, 2)
    axb.plot(DEPTHS, [no_opt[d]["total"] for d in DEPTHS], "o-", color="tab:blue",
             label=Lbl("光学なし (距離+IMU+深度)", "no-optical"))
    if opt_coastal:
        ds = sorted(opt_coastal)
        axb.plot(ds, [opt_coastal[d] for d in ds], "s--", color="tab:green",
                 label=Lbl("光学あり coastal", "optical coastal"))
    cols = {0.05: "skyblue", 0.3: "gold", 1.0: "salmon"}
    for c in DEEP_CLARITIES:
        axb.axvline(det_limits[c], color=cols.get(c, "gray"), ls=":",
                    label=Lbl("%s 検出限界" % CLARITY_LABEL.get(c, "c"),
                              "%s det.limit" % CLARITY_LABEL.get(c, "c")))
    axb.set_xlabel(Lbl("水深 [m]", "depth [m]")); axb.set_ylabel("RMSE total [mm]")
    axb.set_title(Lbl("(b) RMSE vs 水深: 光学なしは濁り非依存で平坦",
                      "(b) RMSE vs depth"))
    axb.grid(alpha=0.3); axb.legend(fontsize=7)

    # (c) 光学なし per-axis vs 水深
    axc = fig.add_subplot(1, 3, 3)
    for k, col in [("x", "tab:orange"), ("y", "tab:green"), ("z", "tab:blue"),
                   ("total", "k")]:
        axc.plot(DEPTHS, [no_opt[d][k] for d in DEPTHS], "o-", color=col,
                 label=k)
    axc.set_xlabel(Lbl("水深 [m]", "depth [m]")); axc.set_ylabel("RMSE [mm]")
    axc.set_title(Lbl("(c) 光学なし 軸別RMSE: zは深度で締まる",
                      "(c) no-optical per-axis"))
    axc.grid(alpha=0.3); axc.legend(fontsize=8)

    fig.suptitle(Lbl(
        "光学なしフォールバック (距離+IMU+深度): 光を使わず濁り・深さの光学劣化に不感、"
        "検出限界の先でも測位継続 (MATH_SPEC §11)",
        "No-optical fallback: range+IMU+depth keeps working past the optical detection limit"))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "no_optical.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "no_optical_rmse_mm": {str(d): no_opt[d] for d in DEPTHS},
        "optical_coastal_total_mm": {str(d): opt_coastal[d] for d in opt_coastal},
        "detection_limit_m": {CLARITY_LABEL.get(c, "c%.2f" % c): det_limits[c]
                              for c in DEEP_CLARITIES},
        "demo": {"depth_m": DEMO_DEPTH, "rmse_mm":
                 {k: rmse_demo[k] * 1000 for k in ("x", "y", "z", "total")}},
    }
    jpath = write_json("run_no_optical", payload,
                       meta={"seed": int(SEED), "n_seeds": N_SEEDS,
                             "sigma_depth_m": SIGMA_DEPTH, "script": "run_no_optical.py"})
    cpath = write_csv("run_no_optical",
                      [{"depth_m": d, "no_optical_total_mm": round(no_opt[d]["total"], 1),
                        "no_optical_z_mm": round(no_opt[d]["z"], 1)} for d in DEPTHS],
                      header=["depth_m", "no_optical_total_mm", "no_optical_z_mm"])
    print(f"\n図   : {png}\nJSON : {jpath}\nCSV  : {cpath}")
    print("\n完了。光学なし(距離+IMU+深度)が濁り・深さに不感で測位を継続できることを確認。")


if __name__ == "__main__":
    main()
