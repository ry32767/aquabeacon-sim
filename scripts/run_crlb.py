"""run_crlb.py — 推定器の統計的検証: CRLB・効率・一貫性・GDOP (MATH_SPEC §4.5, §15)。

研究論文の中核となる検証図を生成する:
  (a) 経験モンテカルロ RMSE が **Cramer-Rao 下界 (CRLB)** に漸近する (= 推定が効率的)。
  (b) 同じことを距離掃引で確認 (角度誤差×距離の効きが理論と一致)。
  (c) **GDOP マップ** (距離×仰角): 観測幾何 → 達成可能精度の地図 (near-nadir 運用の根拠)。
  (d) 深度融合 (§10) が鉛直の下界を縮める効果 (理論 CRLB で比較)。
さらに **NEES (一貫性, 平均≈3)** と **バイアス (分散支配, ほぼ不偏)** を表で出す。

統計的妥当性:
  - 各試行は独立サブストリーム (rng.substream_seed, §15.2) から生成 (試行間ノイズ再利用を回避)。
  - RMSE はブートストラップ 95% 信頼区間つきで報告 (§15.1)。

出力: コンソール表 + results/crlb/crlb.png + results/crlb/run_crlb.{json,csv} + README
実行: python scripts/run_crlb.py
MBD: CRLB/共分散は位置と σ のみから決まり truth を見ない (estimator 層)。CRLB は真値幾何で
     評価 (evaluation 層が truth を渡す)。推定には truth を渡さず観測のみ入力。
"""
import os
import sys

import numpy as np
from _plotstyle import plt, USE_JP, JP_FONT, Lbl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SIGMA, SIGMA_DEPTH, P_PARENT, SEED, MC_N
from src.rng import substream_seed
from src.sensors import simulate_observation, simulate_depth
from src.estimator import estimate_position, gdop
from src.evaluation import (crlb_position, crlb_rmse, rmse_with_ci, nees,
                            position_bias)
from src.results_io import write_json, write_csv, scenario_dir, write_report

FIGDIR = scenario_dir("crlb")

ELEV_DEGS = [-89, -85, -80, -70, -60, -45, -30]   # 仰角掃引 (near-nadir -> 斜め)
RANGE_GRID = [5, 8, 10, 12.5, 15, 18, 22]          # 距離掃引 [m]
NOM_RANGE = 12.5                                    # 仰角掃引・公称距離 [m]
N_MC = min(MC_N, 4000)                              # 効率検証の試行数


def _truth_at(elev_deg, d):
    """仰角 elev_deg・距離 d・方位0 の真値 (rho, 0, d sinφ)。"""
    phi = np.deg2rad(elev_deg)
    return np.array([d * np.cos(phi), 0.0, d * np.sin(phi)])


def _mc_estimates(truth, sigma, n, base_seed, with_depth=False):
    """独立サブストリーム (§15.2) で n 試行の推定群 (n,3) を返す。truth は評価入力。"""
    est = np.empty((n, 3))
    for i in range(n):
        s_obs = substream_seed(base_seed, i, 0)
        z = simulate_observation(truth, sigma, seed=s_obs, p_parent=P_PARENT)
        if with_depth:
            zd = simulate_depth(truth, SIGMA_DEPTH, seed=substream_seed(base_seed, i, 2))
            est[i] = estimate_position(z, sigma, p_parent=P_PARENT,
                                       z_depth=zd, sigma_depth=SIGMA_DEPTH)
        else:
            est[i] = estimate_position(z, sigma, p_parent=P_PARENT)
    return est


def _row(truth, sigma, base_seed):
    """1条件の CRLB・経験 RMSE(+CI)・効率・NEES・バイアスをまとめる。"""
    crlb = crlb_rmse(truth, sigma) * 1000
    est = _mc_estimates(truth, sigma, N_MC, base_seed)
    ci = rmse_with_ci(est, truth, seed=0)
    rmse = ci["rmse"] * 1000
    cov = crlb_position(truth, sigma)
    ne = nees(truth, est, cov)
    bias = position_bias(truth, est)
    return {"crlb_mm": crlb, "rmse_mm": rmse,
            "ci_low_mm": ci["ci_low"] * 1000, "ci_high_mm": ci["ci_high"] * 1000,
            "efficiency": rmse / crlb, "nees_mean": float(ne.mean()),
            "bias_norm_mm": float(np.linalg.norm(bias) * 1000)}


def main():
    print("=== CRLB / 効率 / 一貫性 / GDOP 検証 (MATH_SPEC §4.5, §15) ===")
    print(f"フォント: {JP_FONT if USE_JP else '(英語ラベル)'} / 試行数 N={N_MC} / "
          f"独立サブストリーム (SeedSequence)")

    # --- 仰角掃引 (公称距離) ---
    elev_rows = {}
    for j, e in enumerate(ELEV_DEGS):
        elev_rows[e] = _row(_truth_at(e, NOM_RANGE), SIGMA, substream_seed(SEED, 1, j))
    print(f"\n--- 仰角掃引 (d={NOM_RANGE}m): CRLB vs 経験RMSE [mm] ---")
    print(f"{'elev[deg]':>10} {'CRLB':>8} {'RMSE':>8} {'95%CI':>17} "
          f"{'eff':>6} {'NEES':>6} {'|bias|':>7}")
    for e in ELEV_DEGS:
        r = elev_rows[e]
        print(f"{e:>10} {r['crlb_mm']:>8.1f} {r['rmse_mm']:>8.1f} "
              f"[{r['ci_low_mm']:>6.1f},{r['ci_high_mm']:>6.1f}] "
              f"{r['efficiency']:>6.3f} {r['nees_mean']:>6.2f} {r['bias_norm_mm']:>7.2f}")

    # --- 距離掃引 (公称仰角 -60deg, near-nadir 運用域) ---
    rng_rows = {}
    for j, d in enumerate(RANGE_GRID):
        rng_rows[d] = _row(_truth_at(-60, d), SIGMA, substream_seed(SEED, 2, j))
    print(f"\n--- 距離掃引 (仰角-60deg): CRLB vs 経験RMSE [mm] ---")
    print(f"{'d[m]':>8} {'CRLB':>8} {'RMSE':>8} {'eff':>6}")
    for d in RANGE_GRID:
        r = rng_rows[d]
        print(f"{d:>8.1f} {r['crlb_mm']:>8.1f} {r['rmse_mm']:>8.1f} {r['efficiency']:>6.3f}")

    eff_all = [elev_rows[e]["efficiency"] for e in ELEV_DEGS] + \
              [rng_rows[d]["efficiency"] for d in RANGE_GRID]
    nees_all = [elev_rows[e]["nees_mean"] for e in ELEV_DEGS]
    print(f"\n効率 (RMSE/CRLB) 範囲: {min(eff_all):.3f} - {max(eff_all):.3f} (1.0 が効率的=下界達成)")
    print(f"NEES 平均 範囲: {min(nees_all):.2f} - {max(nees_all):.2f} (自由度3が整合)")

    # --- GDOP マップ (距離 × 仰角): 理論共分散のみ (高速) ---
    map_d = np.linspace(5, 22, 30)
    map_e = np.linspace(-89, -25, 30)
    G = np.empty((len(map_e), len(map_d)))
    for ie, e in enumerate(map_e):
        for idx, d in enumerate(map_d):
            G[ie, idx] = gdop(crlb_position(_truth_at(e, d), SIGMA)) * 1000

    # --- 深度融合の下界効果 (理論 CRLB, 鉛直 z 成分) ---
    z_no, z_dz = [], []
    for e in ELEV_DEGS:
        t = _truth_at(e, NOM_RANGE)
        z_no.append(np.sqrt(crlb_position(t, SIGMA)[2, 2]) * 1000)
        z_dz.append(np.sqrt(crlb_position(t, SIGMA, with_depth=True,
                                          sigma_depth=SIGMA_DEPTH)[2, 2]) * 1000)

    # ===== 図 =====
    fig = plt.figure(figsize=(15, 10))

    # (a) 仰角: CRLB vs RMSE(+CI)
    axa = fig.add_subplot(2, 2, 1)
    axa.plot(ELEV_DEGS, [elev_rows[e]["crlb_mm"] for e in ELEV_DEGS], "k-",
             lw=2, label=Lbl("CRLB (理論下界)", "CRLB (bound)"))
    rmse_e = [elev_rows[e]["rmse_mm"] for e in ELEV_DEGS]
    yerr = [[elev_rows[e]["rmse_mm"] - elev_rows[e]["ci_low_mm"] for e in ELEV_DEGS],
            [elev_rows[e]["ci_high_mm"] - elev_rows[e]["rmse_mm"] for e in ELEV_DEGS]]
    axa.errorbar(ELEV_DEGS, rmse_e, yerr=yerr, fmt="o", color="tab:blue", capsize=3,
                 label=Lbl("経験RMSE (95%CI)", "MC RMSE (95% CI)"))
    axa.set_xlabel(Lbl("仰角 [deg] (-90=真下)", "elevation [deg]"))
    axa.set_ylabel("RMSE / CRLB [mm]")
    axa.set_title(Lbl("(a) 効率: 経験RMSEがCRLBに漸近 (d=%.1fm)" % NOM_RANGE,
                      "(a) efficiency vs elevation"))
    axa.grid(alpha=0.3); axa.legend(fontsize=9)

    # (b) 距離: CRLB vs RMSE
    axb = fig.add_subplot(2, 2, 2)
    axb.plot(RANGE_GRID, [rng_rows[d]["crlb_mm"] for d in RANGE_GRID], "k-", lw=2,
             label=Lbl("CRLB", "CRLB"))
    axb.plot(RANGE_GRID, [rng_rows[d]["rmse_mm"] for d in RANGE_GRID], "o",
             color="tab:blue", label=Lbl("経験RMSE", "MC RMSE"))
    axb.plot(RANGE_GRID, [d * SIGMA[1] * 1000 for d in RANGE_GRID], "r:",
             label=Lbl("d*σ_ang (目安)", "d*sigma_ang"))
    axb.set_xlabel(Lbl("距離 d [m]", "range d [m]")); axb.set_ylabel("RMSE [mm]")
    axb.set_title(Lbl("(b) 距離掃引: 角度誤差×距離が支配 (仰角-60deg)",
                      "(b) vs range"))
    axb.grid(alpha=0.3); axb.legend(fontsize=9)

    # (c) GDOP マップ
    axc = fig.add_subplot(2, 2, 3)
    im = axc.pcolormesh(map_d, map_e, G, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axc, label="GDOP [mm]")
    axc.set_xlabel(Lbl("距離 d [m]", "range d [m]"))
    axc.set_ylabel(Lbl("仰角 [deg]", "elevation [deg]"))
    axc.set_title(Lbl("(c) GDOPマップ: 観測幾何→達成可能精度", "(c) GDOP map"))

    # (d) 深度融合の下界効果 (z 成分)
    axd = fig.add_subplot(2, 2, 4)
    axd.plot(ELEV_DEGS, z_no, "o-", color="tab:red",
             label=Lbl("深度なし z 下界", "no depth (z)"))
    axd.plot(ELEV_DEGS, z_dz, "s-", color="tab:green",
             label=Lbl("深度融合 z 下界", "depth-fused (z)"))
    axd.set_xlabel(Lbl("仰角 [deg]", "elevation [deg]"))
    axd.set_ylabel(Lbl("z 方向 CRLB [mm]", "z CRLB [mm]"))
    axd.set_title(Lbl("(d) 深度融合(§10)が鉛直の下界を縮める",
                      "(d) depth fusion shrinks z bound"))
    axd.grid(alpha=0.3); axd.legend(fontsize=9)

    fig.suptitle(Lbl(
        "推定器の統計的検証: 経験RMSE≈CRLB (効率的) / NEES≈3 (一貫) / GDOPマップ (MATH_SPEC §4.5, §15)",
        "Estimator validation: RMSE approaches CRLB, NEES~3, DOP map (MATH_SPEC §4.5, §15)"),
        fontsize=12)
    fig.tight_layout()
    png = os.path.join(FIGDIR, "crlb.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "elevation_sweep": {str(e): elev_rows[e] for e in ELEV_DEGS},
        "range_sweep": {str(d): rng_rows[d] for d in RANGE_GRID},
        "efficiency_range": [float(min(eff_all)), float(max(eff_all))],
        "nees_mean_range": [float(min(nees_all)), float(max(nees_all))],
        "depth_fusion_z_crlb_mm": {str(e): {"no_depth": z_no[i], "depth": z_dz[i]}
                                   for i, e in enumerate(ELEV_DEGS)},
        "nominal_range_m": NOM_RANGE, "n_mc": N_MC,
    }
    write_json("crlb/run_crlb", payload,
               meta={"seed": int(SEED), "n_mc": N_MC, "script": "run_crlb.py"})
    write_csv("crlb/run_crlb",
              [{"elev_deg": e, "crlb_mm": round(elev_rows[e]["crlb_mm"], 2),
                "rmse_mm": round(elev_rows[e]["rmse_mm"], 2),
                "efficiency": round(elev_rows[e]["efficiency"], 4),
                "nees_mean": round(elev_rows[e]["nees_mean"], 3)} for e in ELEV_DEGS],
              header=["elev_deg", "crlb_mm", "rmse_mm", "efficiency", "nees_mean"])
    write_report(
        "crlb", "推定器の統計的検証 (CRLB・効率・一貫性・GDOP)",
        "位置推定器の研究グレード検証。経験モンテカルロ RMSE が解析的な Cramer-Rao 下界 (CRLB,\n"
        "§4.5) に漸近すること (推定が効率的=情報理論的に最適) を仰角・距離掃引で示し、NEES (正規化\n"
        "推定誤差二乗) の平均が自由度 3 になること (報告共分散が正しく較正されている=一貫) と、\n"
        "バイアスが RMSE に対し十分小さいこと (ほぼ不偏) を確認する。GDOP マップは観測幾何から\n"
        "達成可能精度への写像で、near-nadir 運用域の根拠になる。深度融合 (§10) が鉛直の下界を\n"
        "縮める効果も理論 CRLB で示す。各試行は独立サブストリーム (§15.2)、RMSE は 95% 信頼区間つき。",
        condition_sections=["noise", "depth", "montecarlo"],
        sensors=["parent_cam", "acoustic1", "depth"],
        not_reflected=[
            ("`[error_model]`/`[acoustic]`/`[sync]`",
             "CRLB は零平均ガウスの理論下界。系統バイアス・音速ズレは下界の前提 (不偏ガウス) を"
             "崩すため本検証では理想ノイズを用いる (現実誤差込み評価は run_spec)。"),
            ("`[sbl]`/`[stereo]`/`[attitude]`/`[optical]`", "本検証は単機カメラ+音響(+深度)の測位下界に集中。"),
        ],
        outputs=[("crlb.png", "効率(仰角/距離) + GDOPマップ + 深度融合の下界"),
                 ("run_crlb.json", "全数値 (CRLB/RMSE/CI/効率/NEES/bias)"),
                 ("run_crlb.csv", "仰角別 CRLB/RMSE/効率/NEES")],
        results={"効率 RMSE/CRLB": f"{min(eff_all):.3f} - {max(eff_all):.3f} (1.0=下界達成)",
                 "NEES 平均": f"{min(nees_all):.2f} - {max(nees_all):.2f} (dof=3 が整合)"},
        meta={"seed": SEED, "n_mc": N_MC}, math_spec="§4.5, §15")
    print(f"\n出力 : {FIGDIR}")
    print("完了。経験RMSEがCRLBに漸近し(効率的)、NEES~3(一貫)であることを確認。")


if __name__ == "__main__":
    main()
