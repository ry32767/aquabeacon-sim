"""run_depth.py — 深度センサ融合のデモ (MATH_SPEC §10)。

子機の圧力センサ (深度) を光学×音響に第4の観測として融合する効果を示す:
  (a) 鉛直 z 軸の精度が劇的に向上 (深度は z を直接・濁り非依存で拘束)
  (b) 深い水深で光学角度が悪化しても、深度ありは z と総合RMSEを抑える
  (c) 冗長性 (観測4>未知数3) により単時刻でもロバスト推定が外れ値を棄却できる

(a)(b) は光学リンク減衰 (§9) の σ_ang(d) を「真のノイズ かつ 推定の重み」に使う
well-calibrated 推定 (距離・濁りに応じて角度の信頼度を下げる適応重み) を仮定する。
出力: コンソール表 + figures/depth/depth_fusion.png + results/run_depth.{json,csv}

実行: python scripts/run_depth.py
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

from src.config import (SIGMA, P_PARENT, SEED, OPTICAL_MODEL, SIGMA_DEPTH,
                        DEPTH_BIAS, DEEP_DEPTHS, DEEP_HORIZ_OFFSET, DEEP_MC_N)
from src.sensors import simulate_observation, simulate_depth, optical_angular_sigma
from src.estimator import estimate_position
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_csv, scenario_dir, write_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = scenario_dir("depth")
_JP_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "Hiragino Sans", "TakaoPGothic", "IPAexGothic"]
_available = {f.name for f in fm.fontManager.ttflist}
_JP = next((c for c in _JP_CANDIDATES if c in _available), None)
USE_JP = _JP is not None
if USE_JP:
    plt.rcParams["font.family"] = _JP
plt.rcParams["axes.unicode_minus"] = False

CLARITY = 0.3                # 掃引に使う濁り (coastal)
PANEL_A_DEPTH = 15.0         # (a) per-axis を見る水深
OUTLIER_TRUTH = np.array([8.0, 6.0, -5.0])   # (c) 単時刻: 過度に急峻でない幾何


def Lbl(ja, en):
    return ja if USE_JP else en


def _truth_at_depth(depth):
    return np.array([DEEP_HORIZ_OFFSET[0], DEEP_HORIZ_OFFSET[1], -float(depth)])


def _sigma_at_depth(depth, clarity):
    """光学減衰モデルから、その水深での角度ノイズ σ_ang(d) を反映した σ を返す。"""
    m = dict(OPTICAL_MODEL)
    m["attenuation_c"] = clarity
    r = float(np.linalg.norm(_truth_at_depth(depth)))
    s_ang = optical_angular_sigma(r, m)
    return (SIGMA[0], s_ang, s_ang)


def _rmse_axes(depth, clarity, use_depth, seed=SEED, n=DEEP_MC_N):
    """well-calibrated 推定で深度あり/なしの per-axis RMSE [mm] を返す。"""
    truth = _truth_at_depth(depth)
    sig = _sigma_at_depth(depth, clarity)
    est = np.empty((n, 3))
    for i in range(n):
        z = simulate_observation(truth, sig, seed=seed + i, p_parent=P_PARENT)
        if use_depth:
            zd = simulate_depth(truth, SIGMA_DEPTH, seed=seed + 500000 + i,
                                bias=DEPTH_BIAS)
            est[i] = estimate_position(z, sig, p_parent=P_PARENT,
                                       z_depth=zd, sigma_depth=SIGMA_DEPTH)
        else:
            est[i] = estimate_position(z, sig, p_parent=P_PARENT)
    r = rmse_xyz(truth, est)
    return {k: r[k] * 1000 for k in ("x", "y", "z", "total")}


def panel_single_time_outlier(seed=SEED, n=DEEP_MC_N):
    """仰角 φ に外れ値を常時注入し、3手法の位置RMSE [mm] を返す (単時刻・冗長性)。"""
    truth = OUTLIER_TRUTH
    errs = {"L2(深度なし)": [], "L2(深度あり)": [], "huber(深度あり)": []}
    for i in range(n):
        z = simulate_observation(truth, SIGMA, seed=seed + i, p_parent=P_PARENT).copy()
        z[2] += np.deg2rad(12.0)                  # 仰角に外れ値 (誤対応相当)
        zd = simulate_depth(truth, SIGMA_DEPTH, seed=seed + 900000 + i)
        e1 = estimate_position(z, SIGMA, p_parent=P_PARENT)
        e2 = estimate_position(z, SIGMA, p_parent=P_PARENT, z_depth=zd,
                               sigma_depth=SIGMA_DEPTH)
        e3 = estimate_position(z, SIGMA, p_parent=P_PARENT, loss="huber",
                               z_depth=zd, sigma_depth=SIGMA_DEPTH)
        errs["L2(深度なし)"].append(np.linalg.norm(e1 - truth) * 1000)
        errs["L2(深度あり)"].append(np.linalg.norm(e2 - truth) * 1000)
        errs["huber(深度あり)"].append(np.linalg.norm(e3 - truth) * 1000)
    return {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}


def main(seed=SEED):
    print("=== 深度センサ融合デモ (MATH_SPEC §10) ===")
    print(f"フォント: {_JP if USE_JP else '(英語ラベル)'} / σ_depth={SIGMA_DEPTH*100:.0f} cm "
          f"/ MC={DEEP_MC_N}")

    # (a) per-axis RMSE
    a_no = _rmse_axes(PANEL_A_DEPTH, CLARITY, use_depth=False)
    a_dp = _rmse_axes(PANEL_A_DEPTH, CLARITY, use_depth=True)
    print(f"\n--- (a) 深さ{PANEL_A_DEPTH:.0f}m (c={CLARITY}) per-axis RMSE [mm] ---")
    print(f"  {'':10s}    x      y      z    total")
    print(f"  深度なし   {a_no['x']:6.0f} {a_no['y']:6.0f} {a_no['z']:6.0f} {a_no['total']:7.0f}")
    print(f"  深度あり   {a_dp['x']:6.0f} {a_dp['y']:6.0f} {a_dp['z']:6.0f} {a_dp['total']:7.0f}")
    print(f"  -> z軸 RMSE {a_no['z']:.0f} -> {a_dp['z']:.0f} mm")

    # (b) total & z RMSE vs depth
    depths = DEEP_DEPTHS
    tot_no, tot_dp, z_no, z_dp = [], [], [], []
    for d in depths:
        rn = _rmse_axes(d, CLARITY, use_depth=False)
        rd = _rmse_axes(d, CLARITY, use_depth=True)
        tot_no.append(rn["total"]); tot_dp.append(rd["total"])
        z_no.append(rn["z"]); z_dp.append(rd["z"])
    print(f"\n--- (b) z軸RMSE vs 水深 (c={CLARITY}) [mm] ---")
    print("  水深[m] " + "".join("%7.0f" % d for d in depths))
    print("  z なし  " + "".join("%7.0f" % v for v in z_no))
    print("  z あり  " + "".join("%7.0f" % v for v in z_dp))

    # (c) single-time outlier
    c_errs = panel_single_time_outlier(seed=seed)
    print("\n--- (c) 単時刻・仰角外れ値下の位置RMSE [mm] ---")
    for k, v in c_errs.items():
        print(f"  {k:16s} {v:7.0f}")

    # ---- 図 ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    labels = ["x", "y", "z", "total"]
    xpos = np.arange(len(labels))
    axs[0].bar(xpos - 0.2, [a_no[k] for k in labels], 0.4,
               label=Lbl("深度なし", "no depth"), color="gray")
    axs[0].bar(xpos + 0.2, [a_dp[k] for k in labels], 0.4,
               label=Lbl("深度あり", "with depth"), color="tab:blue")
    axs[0].set_xticks(xpos); axs[0].set_xticklabels(labels)
    axs[0].set_ylabel("RMSE [mm]")
    axs[0].set_title(Lbl("(a) 軸別RMSE 深さ%.0fm: z が激減" % PANEL_A_DEPTH,
                         "(a) per-axis RMSE @ %.0fm" % PANEL_A_DEPTH))
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3, axis="y")

    axs[1].plot(depths, z_no, "o-", color="gray", label=Lbl("z 深度なし", "z no depth"))
    axs[1].plot(depths, z_dp, "o-", color="tab:blue", label=Lbl("z 深度あり", "z with depth"))
    axs[1].plot(depths, tot_no, "s--", color="gray", alpha=0.5,
                label=Lbl("総合 なし", "total no"))
    axs[1].plot(depths, tot_dp, "s--", color="tab:blue", alpha=0.5,
                label=Lbl("総合 あり", "total with"))
    axs[1].set_xlabel(Lbl("水深 [m]", "depth [m]")); axs[1].set_ylabel("RMSE [mm]")
    axs[1].set_yscale("log")
    axs[1].set_title(Lbl("(b) z/総合RMSE vs 水深 (c=%.1f)" % CLARITY, "(b) RMSE vs depth"))
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3, which="both")

    keys = list(c_errs.keys())
    axs[2].bar(range(len(keys)), [c_errs[k] for k in keys],
               color=["gray", "tab:cyan", "tab:blue"])
    axs[2].set_xticks(range(len(keys)))
    axs[2].set_xticklabels(keys, rotation=12, fontsize=8)
    axs[2].set_ylabel(Lbl("位置RMSE [mm]", "pos RMSE [mm]"))
    axs[2].set_title(Lbl("(c) 単時刻・仰角外れ値\n冗長性でロバスト棄却",
                         "(c) single-time outlier"))
    axs[2].grid(alpha=0.3, axis="y")

    fig.suptitle(Lbl(
        "深度センサ融合: 鉛直zを直接拘束し、冗長性で単時刻ロバストを可能にする (MATH_SPEC §10)",
        "Depth-sensor fusion: pins vertical z and enables single-time robust rejection"))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "depth_fusion.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "per_axis_panel": {"depth_m": PANEL_A_DEPTH, "clarity_c": CLARITY,
                           "no_depth_mm": a_no, "with_depth_mm": a_dp},
        "vs_depth": {"clarity_c": CLARITY, "depths_m": list(depths),
                     "z_no_depth_mm": z_no, "z_with_depth_mm": z_dp,
                     "total_no_depth_mm": tot_no, "total_with_depth_mm": tot_dp},
        "single_time_outlier_rmse_mm": c_errs,
    }
    jpath = write_json("depth/run_depth", payload,
                       meta={"seed": int(seed), "sigma_depth_m": SIGMA_DEPTH,
                             "mc_n": int(DEEP_MC_N), "script": "run_depth.py"})
    cpath = write_csv("depth/run_depth",
                      [{"depth_m": d, "z_no_depth_mm": round(a, 1),
                        "z_with_depth_mm": round(b, 1)}
                       for d, a, b in zip(depths, z_no, z_dp)],
                      header=["depth_m", "z_no_depth_mm", "z_with_depth_mm"])
    write_report(
        "depth", "深度センサ融合デモ",
        "子機の圧力センサ (深度=-z) を光学×音響に第4の観測として融合する効果を示す。\n"
        "(a) 鉛直 z 軸の精度が劇的向上 (深度は z を直接・濁り非依存で拘束)、(b) 深い水深で角度が\n"
        "悪化しても深度ありは z と総合RMSEを抑える、(c) 冗長性 (観測4>未知数3) で単時刻でも\n"
        "ロバスト推定が外れ値を棄却できる。光学σは減衰モデル(§9)で校正した適応重みを仮定。",
        condition_sections=["noise", "depth", "optical", "deepwater"],
        outputs=[("depth_fusion.png", "軸別RMSE/vs水深/単時刻外れ値の3パネル"),
                 ("run_depth.json", "全結果"),
                 ("run_depth.csv", "水深別 z RMSE (深度あり/なし)")],
        results={f"深さ{PANEL_A_DEPTH:.0f}m z RMSE (なし→あり)":
                 f"{a_no['z']:.0f} → {a_dp['z']:.0f} mm",
                 "単時刻外れ値 L2なし→huber深度あり":
                 f"{c_errs['L2(深度なし)']:.0f} → {c_errs['huber(深度あり)']:.0f} mm"},
        meta={"seed": seed, "sigma_depth_cm": SIGMA_DEPTH * 100}, math_spec="§10")
    print(f"\n出力 : {FIGDIR}")
    print("\n完了。深度センサで z 精度が激増し、冗長性で単時刻ロバストが効くことを確認。")


if __name__ == "__main__":
    main()
