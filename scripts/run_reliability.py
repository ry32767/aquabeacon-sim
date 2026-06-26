"""run_reliability.py — 推定器の信頼性掃引 (故障率・分布) と新誤差モデルへのロバスト性 (MATH_SPEC §15)。

平均 RMSE は稀な発散を埋もれさせる。本シナリオは**分布**で信頼性を示す:
  (a) 外れ値率 × 損失 (L2 vs Huber): 粗大故障率と p95 誤差。冗長(IMU+深度)なロバスト推定が
      外れ値下で発散を抑える (MATH_SPEC §4.4, §8.3)。
  (b) 時間相関ノイズ ρ (§8.6): 白色のみだと過大評価。ρ↑ で RMSE が現実的に悪化することを正直に示す。
  (c) IMU 変位バイアス σ_imu_bias (§5.5): 光学なしフォールバックの精度がバイアスで劣化することを示す。

統計的妥当性: 各試行は独立サブストリーム (rng.substream_seed, §15.2)。中央値・p95・故障率を報告。
出力: コンソール表 + results/reliability/reliability.png + run_reliability.{json,csv} + README
実行: python scripts/run_reliability.py
MBD: 推定には truth を渡さず観測のみ。評価でだけ真値と突き合わせる。
"""
import os
import sys

import numpy as np
from _plotstyle import plt, USE_JP, JP_FONT, Lbl as L

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_IMU, SIGMA_DEPTH, DEPTH_BIAS, P_PARENT, SEED,
                        SURVEY_AREA, SURVEY_ORIGIN, ERROR_MODEL, N_SEEDS_TRAJ)
from src.rng import substream_seed
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_observation_sequence,
                         simulate_observation_sequence_realistic,
                         simulate_imu_displacements, simulate_depth_sequence)
from src.estimator import estimate_trajectory, estimate_trajectory_acoustic_inertial
from src.evaluation import rmse_xyz, error_distribution_stats
from src.results_io import write_json, write_csv, scenario_dir, write_report

FIGDIR = scenario_dir("reliability")

DEPTH = 12.0
N_TRIALS = max(N_SEEDS_TRAJ, 40)        # 故障率は裾を見るので多めに
FAIL_MM = 300.0                         # 粗大故障しきい値 [mm]
OUTLIER_RATES = [0.0, 0.05, 0.1, 0.2, 0.3]
RHOS = [0.0, 0.3, 0.6, 0.9]
IMU_BIASES = [0.0, 0.001, 0.002, 0.005]


def _traj():
    return double_lawnmower_trajectory(area=SURVEY_AREA, depth=-DEPTH, n_legs=2,
                                       pts_per_leg=6, origin=SURVEY_ORIGIN)


def _trial_rmse_outlier(seed, outlier_rate, loss):
    """外れ値率つき観測 (現実誤差) で軌道推定し total RMSE [mm] を返す (IMU+深度, ロバスト損失)。"""
    traj = _traj()
    err = dict(ERROR_MODEL)                     # 音速ズレ等の現実誤差 + 外れ値率
    err["outlier_rate"] = outlier_rate
    z = simulate_observation_sequence_realistic(
        traj, SIGMA, seed=substream_seed(seed, 0), p_parent=P_PARENT, **err)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=substream_seed(seed, 1))
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=substream_seed(seed, 2),
                                  bias=DEPTH_BIAS)
    est = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                              p_parent=P_PARENT, z_depth_seq=dep,
                              sigma_depth=SIGMA_DEPTH, loss=loss)
    return rmse_xyz(traj, est)["total"] * 1000


def _trial_rmse_rho(seed, rho):
    """時間相関 ρ の白色化光学観測で軌道推定 (IMU+深度)。"""
    traj = _traj()
    z = simulate_observation_sequence(traj, SIGMA, seed=substream_seed(seed, 0),
                                      p_parent=P_PARENT, rho=rho)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=substream_seed(seed, 1))
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=substream_seed(seed, 2),
                                  bias=DEPTH_BIAS, rho=rho)
    est = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                              p_parent=P_PARENT, z_depth_seq=dep, sigma_depth=SIGMA_DEPTH)
    return rmse_xyz(traj, est)["total"] * 1000


def _trial_rmse_imubias(seed, sigma_bias):
    """IMU 変位バイアス付きで**光学なしフォールバック** (距離+IMU+深度) を推定。"""
    traj = _traj()
    z = simulate_observation_sequence(traj, SIGMA, seed=substream_seed(seed, 0),
                                      p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=substream_seed(seed, 1),
                                     sigma_bias=sigma_bias)
    dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=substream_seed(seed, 2),
                                  bias=DEPTH_BIAS)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimate_trajectory_acoustic_inertial(z[:, 0], SIGMA[0], imu, SIGMA_IMU,
                                                    dep, SIGMA_DEPTH, p_parent=P_PARENT)
    return rmse_xyz(traj, est)["total"] * 1000


def _sweep(fn, values, label):
    """各値で N_TRIALS 独立試行 -> error_distribution_stats のリストを返す。"""
    rows = []
    for vi, v in enumerate(values):
        errs = [fn(substream_seed(SEED, vi, s), v) for s in range(N_TRIALS)]
        st = error_distribution_stats(errs, FAIL_MM)
        rows.append(st)
        print(f"  {label}={v:<6}: median {st['median']:6.0f}  p95 {st['p95']:6.0f}  "
              f"mean {st['mean']:6.0f}  fail {st['fail_rate']*100:5.1f}%")
    return rows


def main():
    print("=== 推定器の信頼性掃引 (故障率・分布 + 新誤差モデルロバスト性) (MATH_SPEC §15) ===")
    print(f"フォント: {JP_FONT if USE_JP else '(英語ラベル)'} / 深さ{DEPTH:.0f}m / "
          f"独立{N_TRIALS}試行 / 故障しきい値 {FAIL_MM:.0f}mm")

    print("\n--- (a) 外れ値率 × L2 ---")
    l2 = _sweep(lambda s, v: _trial_rmse_outlier(s, v, "linear"), OUTLIER_RATES, "p")
    print("--- (a) 外れ値率 × Huber (ロバスト) ---")
    hub = _sweep(lambda s, v: _trial_rmse_outlier(s, v, "huber"), OUTLIER_RATES, "p")
    print("\n--- (b) 時間相関ノイズ ρ (§8.6) ---")
    rho = _sweep(_trial_rmse_rho, RHOS, "rho")
    print("\n--- (c) IMU 変位バイアス σ_bias (§5.5, 光学なし) ---")
    bias = _sweep(_trial_rmse_imubias, IMU_BIASES, "bias")

    # ===== 図 =====
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) 故障率 + p95: L2 vs Huber
    axs[0].plot(OUTLIER_RATES, [r["fail_rate"] * 100 for r in l2], "o-",
                color="tab:red", label=L("L2 故障率", "L2 fail%"))
    axs[0].plot(OUTLIER_RATES, [r["fail_rate"] * 100 for r in hub], "s-",
                color="tab:green", label=L("Huber 故障率", "Huber fail%"))
    axs[0].set_xlabel(L("外れ値率 p", "outlier rate p"))
    axs[0].set_ylabel(L("粗大故障率 [%%] (>%.0fmm)" % FAIL_MM, "fail rate [%]"))
    axs[0].set_title(L("(a) 外れ値下の故障率: ロバストが発散を抑える",
                       "(a) fail rate vs outliers"))
    axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
    ax0b = axs[0].twinx()
    ax0b.plot(OUTLIER_RATES, [r["p95"] for r in l2], "o:", color="salmon", alpha=0.7)
    ax0b.plot(OUTLIER_RATES, [r["p95"] for r in hub], "s:", color="lightgreen", alpha=0.7)
    ax0b.set_ylabel(L("p95 誤差 [mm] (点線)", "p95 [mm] (dotted)"))

    # (b) RMSE vs rho
    axs[1].plot(RHOS, [r["median"] for r in rho], "o-", color="tab:blue",
                label=L("中央値", "median"))
    axs[1].fill_between(RHOS, [r["median"] for r in rho], [r["p95"] for r in rho],
                        color="tab:blue", alpha=0.15, label=L("中央値→p95", "median→p95"))
    axs[1].set_xlabel(L("時間相関 ρ (§8.6)", "temporal corr. ρ"))
    axs[1].set_ylabel("RMSE total [mm]")
    axs[1].set_title(L("(b) 時間相関ノイズで精度が現実的に劣化",
                       "(b) colored noise degrades RMSE"))
    axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)

    # (c) RMSE vs IMU bias (fallback)
    axs[2].plot([b * 1000 for b in IMU_BIASES], [r["median"] for r in bias], "o-",
                color="tab:purple", label=L("中央値", "median"))
    axs[2].fill_between([b * 1000 for b in IMU_BIASES], [r["median"] for r in bias],
                        [r["p95"] for r in bias], color="tab:purple", alpha=0.15,
                        label=L("中央値→p95", "median→p95"))
    axs[2].set_xlabel(L("IMUバイアス σ_bias [mm/step] (§5.5)", "IMU bias [mm/step]"))
    axs[2].set_ylabel("RMSE total [mm]")
    axs[2].set_title(L("(c) IMUドリフトで光学なし精度が劣化",
                       "(c) IMU bias degrades no-optical"))
    axs[2].grid(alpha=0.3); axs[2].legend(fontsize=8)

    fig.suptitle(L(
        "信頼性掃引: 外れ値の故障率(ロバストで抑制) / 時間相関 / IMUバイアスへの現実的な劣化 (MATH_SPEC §15)",
        "Reliability: fail-rate under outliers (robust helps), colored noise & IMU-bias degradation"))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "reliability.png")
    fig.savefig(png, bbox_inches="tight"); plt.close(fig)

    payload = {
        "fail_threshold_mm": FAIL_MM, "n_trials": N_TRIALS, "depth_m": DEPTH,
        "outlier_L2": {str(p): r for p, r in zip(OUTLIER_RATES, l2)},
        "outlier_huber": {str(p): r for p, r in zip(OUTLIER_RATES, hub)},
        "rho": {str(r): s for r, s in zip(RHOS, rho)},
        "imu_bias": {str(b): s for b, s in zip(IMU_BIASES, bias)},
    }
    write_json("reliability/run_reliability", payload,
               meta={"seed": int(SEED), "n_trials": N_TRIALS, "script": "run_reliability.py"})
    write_csv("reliability/run_reliability",
              [{"outlier_rate": p, "L2_fail_pct": round(l2[i]["fail_rate"] * 100, 1),
                "huber_fail_pct": round(hub[i]["fail_rate"] * 100, 1),
                "L2_p95_mm": round(l2[i]["p95"], 0), "huber_p95_mm": round(hub[i]["p95"], 0)}
               for i, p in enumerate(OUTLIER_RATES)],
              header=["outlier_rate", "L2_fail_pct", "huber_fail_pct", "L2_p95_mm", "huber_p95_mm"])
    write_report(
        "reliability", "推定器の信頼性掃引 (故障率・分布・ロバスト性)",
        "平均 RMSE は稀な発散を埋もれさせる。本シナリオは**分布**(中央値・p95・粗大故障率)で\n"
        "信頼性を示す。(a) 外れ値率を振って L2 と Huber の故障率・p95 を比較 (冗長な IMU+深度の\n"
        "ロバスト推定が発散を抑える, §4.4/§8.3)。(b) 時間相関ノイズ ρ (§8.6) で精度が現実的に\n"
        "劣化することを正直に示す (白色のみは平滑化利得を過大評価)。(c) IMU 変位バイアス (§5.5) で\n"
        "光学なしフォールバックが劣化することを示す。各試行は独立サブストリーム (§15.2)。",
        condition_sections=["survey", "noise", "depth", "error_model", "estimator", "montecarlo"],
        sensors=["parent_cam", "acoustic1", "imu", "depth"],
        not_reflected=[
            ("`[optical]` (減衰σ/見失い)",
             "本シナリオは外れ値・時間相関・IMUバイアスへのロバスト性に集中。光減衰は run_deepwater。"),
            ("`[sbl]`/`[attitude]`/`[stereo]`", "SBL・親機姿勢・ステレオは扱わない (別シナリオ)。"),
        ],
        outputs=[("reliability.png", "故障率(L2 vs Huber)/時間相関/IMUバイアスの劣化"),
                 ("run_reliability.json", "全分布統計"),
                 ("run_reliability.csv", "外れ値率別の故障率・p95")],
        results={"外れ値30% 故障率 (L2 -> Huber)":
                 f"{l2[-1]['fail_rate']*100:.0f}% -> {hub[-1]['fail_rate']*100:.0f}%",
                 "rho=0 -> 0.9 中央値RMSE": f"{rho[0]['median']:.0f} -> {rho[-1]['median']:.0f} mm"},
        meta={"seed": SEED, "n_trials": N_TRIALS}, math_spec="§15, §4.4, §8.6, §5.5")
    print(f"\n出力 : {FIGDIR}")
    print("完了。故障率・p95・時間相関/IMUバイアス劣化を分布で確認。")


if __name__ == "__main__":
    main()
