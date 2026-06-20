"""run_opmap.py — 2次元運用スペック: 濁り×水深の運用可能領域マップ (MATH_SPEC §9-§12)。

水深 (横) × 濁り c (縦) の格子で、各条件をどのモードで運用できるかを塗り分ける:
  - 緑 (optical)  : 光学が信頼でき (見失い確率 <= しきい値) 目標精度を満たす
  - 金 (fallback) : 光学は不可 (見失い多発) だが、距離+IMU+深度のフォールバック (§11) で
                    目標を満たす  ← 自動切替で運用継続できる領域
  - 赤 (none)     : フォールバックでも目標未達 (深すぎ) = 運用不可

光学は濁り・水深で劣化 (§9)、フォールバックは光を使わないので濁り非依存 (§11)。
この地図は自動切替 (§12) の判断境界そのもの。

出力: figures/opmap/operational_map.png + results/run_opmap.{json,csv}
実行: python scripts/run_opmap.py
MBD: 推定には truth を渡さず観測のみ入力。評価でだけ真値と突き合わせる。
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_IMU, SIGMA_DEPTH, P_PARENT, SEED, OPTICAL_MODEL,
                        SPEC_OPDEPTH_TARGET_MM, SWITCH_DROPOUT_THRESHOLD)
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_observation_sequence,
                         simulate_observation_sequence_realistic,
                         simulate_imu_displacements, simulate_depth_sequence,
                         optical_angular_sigma, optical_dropout_prob)
from src.estimator import (estimate_trajectory, estimate_trajectory_acoustic_inertial)
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures", "opmap")
os.makedirs(FIGDIR, exist_ok=True)
_JP_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "Hiragino Sans", "TakaoPGothic", "IPAexGothic"]
_available = {f.name for f in fm.fontManager.ttflist}
_JP = next((c for c in _JP_CANDIDATES if c in _available), None)
USE_JP = _JP is not None
if USE_JP:
    plt.rcParams["font.family"] = _JP
plt.rcParams["axes.unicode_minus"] = False

TARGET = SPEC_OPDEPTH_TARGET_MM         # ミッション精度 [mm]
P_SWITCH = SWITCH_DROPOUT_THRESHOLD     # 見失い確率の切替しきい値
DEPTHS = np.linspace(3.0, 24.0, 12)     # 水深格子 [m]
CLARITIES = np.linspace(0.05, 1.5, 12)  # 濁り格子 [1/m]
N_SEEDS = 3
AZ_STARTS = 8                           # フォールバックの方位スタート数 (速度優先)

OPTICAL, FALLBACK, NONE = 0, 1, 2


def Lbl(ja, en):
    return ja if USE_JP else en


def _traj(depth):
    return double_lawnmower_trajectory(area=(6.0, 4.0), depth=-float(depth),
                                       n_legs=2, pts_per_leg=6, origin=(3.0, 3.0))


def _opt_model(c):
    m = dict(OPTICAL_MODEL); m["attenuation_c"] = c
    return m


def _range(depth):
    return float(np.linalg.norm([3.0, 2.0, -depth]))   # 代表レンジ


def _optical_rmse(depth, clarity):
    """光学あり (減衰§9: 校正σ+見失い) + IMU + 深度, robust の軌道RMSE [mm] 平均。"""
    traj = _traj(depth)
    model = _opt_model(clarity)
    s_ang = optical_angular_sigma(float(np.linalg.norm(traj.mean(axis=0))), model)
    sig = (SIGMA[0], s_ang, s_ang)
    vals = []
    for s in range(N_SEEDS):
        z = simulate_observation_sequence_realistic(
            traj, SIGMA, seed=SEED + s, p_parent=P_PARENT, optical_model=model)
        imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=SEED + s + 100)
        dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=SEED + s + 200)
        est = estimate_trajectory(z, sig, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss="huber",
                                  z_depth_seq=dep, sigma_depth=SIGMA_DEPTH)
        vals.append(rmse_xyz(traj, est)["total"] * 1000)
    return float(np.mean(vals))


def _fallback_rmse(depth):
    """距離+IMU+深度 (光学なし§11) の軌道RMSE [mm] 平均。濁り非依存。"""
    traj = _traj(depth)
    vals = []
    for s in range(N_SEEDS):
        z = simulate_observation_sequence(traj, SIGMA, seed=SEED + s, p_parent=P_PARENT)
        imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=SEED + s + 100)
        dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=SEED + s + 200)
        est = estimate_trajectory_acoustic_inertial(
            z[:, 0], SIGMA[0], imu, SIGMA_IMU, dep, SIGMA_DEPTH,
            p_parent=P_PARENT, n_azimuth_starts=AZ_STARTS)
        vals.append(rmse_xyz(traj, est)["total"] * 1000)
    return float(np.mean(vals))


def build_map():
    # フォールバックは濁り非依存 -> 水深ごとに1回
    fb = {d: _fallback_rmse(d) for d in DEPTHS}
    region = np.empty((len(CLARITIES), len(DEPTHS)), dtype=int)
    pdrop = np.empty_like(region, dtype=float)
    for i, c in enumerate(CLARITIES):
        model = _opt_model(c)
        for j, d in enumerate(DEPTHS):
            p = optical_dropout_prob(_range(d), model)
            pdrop[i, j] = p
            optical_ok = (p <= P_SWITCH)
            if optical_ok and _optical_rmse(d, c) <= TARGET:
                region[i, j] = OPTICAL
            elif fb[d] <= TARGET:
                region[i, j] = FALLBACK
            else:
                region[i, j] = NONE
    return region, pdrop, fb


def main():
    print("=== 2次元運用スペック: 濁り×水深の運用可能領域マップ (§9-§12) ===")
    print(f"フォント: {_JP if USE_JP else '(英語ラベル)'} / 目標 {TARGET:.0f}mm "
          f"/ 切替しきい値 見失い率 {P_SWITCH:.2f} / 平均試行 {N_SEEDS}")

    region, pdrop, fb = build_map()

    # コンソール要約: 各濁りでの (optical最大水深, fallback最大水深)
    print("\n--- 各濁りでの運用可能水深 [m] (optical=緑 / fallback=金) ---")
    print("  濁りc    光学可(最深)  フォールバック可(最深)")
    for i, c in enumerate(CLARITIES):
        opt_d = [DEPTHS[j] for j in range(len(DEPTHS)) if region[i, j] == OPTICAL]
        fb_d = [DEPTHS[j] for j in range(len(DEPTHS)) if region[i, j] in (OPTICAL, FALLBACK)]
        od = max(opt_d) if opt_d else 0.0
        fd = max(fb_d) if fb_d else 0.0
        print(f"  {c:5.2f}      {od:6.1f}        {fd:6.1f}")

    # ===== ヒートマップ =====
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    cmap = ListedColormap(["#2ca02c", "#ffcc33", "#d62728"])   # 緑/金/赤
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    dd = DEPTHS[1] - DEPTHS[0]
    cc = CLARITIES[1] - CLARITIES[0]
    extent = [DEPTHS[0] - dd / 2, DEPTHS[-1] + dd / 2,
              CLARITIES[0] - cc / 2, CLARITIES[-1] + cc / 2]
    ax.imshow(region, origin="lower", aspect="auto", cmap=cmap, norm=norm,
              extent=extent, interpolation="nearest")

    # 見失い率の切替境界 (p_drop = P_SWITCH) を等高線で重ねる
    X, Y = np.meshgrid(DEPTHS, CLARITIES)
    cs = ax.contour(X, Y, pdrop, levels=[P_SWITCH], colors="k",
                    linewidths=1.5, linestyles="--")
    ax.clabel(cs, fmt={P_SWITCH: Lbl("切替境界(見失い%.0f%%)" % (P_SWITCH * 100),
                                     "switch %.0f%%" % (P_SWITCH * 100))}, fontsize=8)

    ax.set_xlabel(Lbl("水深 [m]", "depth [m]"))
    ax.set_ylabel(Lbl("濁り c [1/m] (clear→turbid)", "turbidity c [1/m]"))
    ax.set_title(Lbl(
        "運用可能領域マップ (目標 RMSE <= %.0f mm)\n"
        "緑=光学 / 金=フォールバック(距離+IMU+深度) / 赤=運用不可" % TARGET,
        "Operational map (target %.0f mm)" % TARGET))
    # 凡例
    from matplotlib.patches import Patch
    handles = [Patch(color="#2ca02c", label=Lbl("光学 (optical)", "optical")),
               Patch(color="#ffcc33", label=Lbl("フォールバック (距離+IMU+深度)", "fallback")),
               Patch(color="#d62728", label=Lbl("運用不可 (none)", "none"))]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    png = os.path.join(FIGDIR, "operational_map.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # ===== 出力 =====
    names = {OPTICAL: "optical", FALLBACK: "fallback", NONE: "none"}
    payload = {
        "target_rmse_mm": TARGET, "switch_dropout_threshold": P_SWITCH,
        "depths_m": DEPTHS.tolist(), "clarities": CLARITIES.tolist(),
        "region": region.tolist(), "region_names": names,
        "fallback_rmse_mm": {("%.1f" % d): fb[d] for d in DEPTHS},
    }
    jpath = write_json("run_opmap", payload,
                       meta={"seed": int(SEED), "n_seeds": N_SEEDS,
                             "script": "run_opmap.py"})
    rows = []
    for i, c in enumerate(CLARITIES):
        for j, d in enumerate(DEPTHS):
            rows.append({"clarity_c": round(c, 3), "depth_m": round(d, 1),
                         "region": names[region[i, j]]})
    cpath = write_csv("run_opmap", rows, header=["clarity_c", "depth_m", "region"])
    print(f"\n図   : {png}\nJSON : {jpath}\nCSV  : {cpath}")
    print("\n完了。濁り×水深の運用可能領域を3色で塗り分け。金=フォールバック切替で継続可。")


if __name__ == "__main__":
    main()
