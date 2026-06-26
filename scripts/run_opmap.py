"""run_opmap.py — 2次元運用スペック: 濁り×水深の達成精度マップ (MATH_SPEC §9-§12)。

水深 (横) × 濁り c (縦) の格子で、各条件で**実際に到達できる最良の測位精度 (RMSE)** を
色の濃淡で連続表示する:
  - 見失い率 <= しきい値 : 光学 (角度+距離+IMU+深度) と フォールバック の高精度な方
  - 見失い率 >  しきい値 : 距離+IMU+深度フォールバック (§11) のみ (濁り非依存)
達成RMSE を緑(高精度)→赤(低精度)のグラデーションで描き、太線=目標精度の境界、白破線=切替境界
を重ねる。離散の運用可否 (緑/金/赤) ではなく濃淡にすることで、現実誤差 (§8) と光減衰 (§9) を
反映しても『どのセルでどれだけの精度が出るか』が読み取れる (全面赤で潰れない)。

光学は濁り・水深で劣化 (§9)、フォールバックは光を使わないので濁り非依存 (§11)。

出力: figures/opmap/operational_map.png + results/run_opmap.{json,csv}
実行: python scripts/run_opmap.py
MBD: 推定には truth を渡さず観測のみ入力。評価でだけ真値と突き合わせる。
"""
import os
import sys

import numpy as np
from _plotstyle import plt, USE_JP, JP_FONT, Lbl
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_IMU, SIGMA_DEPTH, DEPTH_BIAS, P_PARENT, SEED,
                        OPTICAL_MODEL, SPEC_OPDEPTH_TARGET_MM,
                        SWITCH_DROPOUT_THRESHOLD, SURVEY_AREA, SURVEY_ORIGIN,
                        ERROR_MODEL, ERROR_MODEL_ENABLE)
from src.rng import substream_seed
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_observation_sequence,
                         simulate_observation_sequence_realistic,
                         simulate_imu_displacements, simulate_depth_sequence,
                         optical_angular_sigma, optical_dropout_prob,
                         apply_attitude_error_config)
from src.estimator import (estimate_trajectory, estimate_trajectory_acoustic_inertial)
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_csv, scenario_dir, write_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = scenario_dir("opmap")

TARGET = SPEC_OPDEPTH_TARGET_MM         # ミッション精度 [mm]
P_SWITCH = SWITCH_DROPOUT_THRESHOLD     # 見失い確率の切替しきい値
DEPTHS = np.linspace(3.0, 24.0, 12)     # 水深格子 [m]
CLARITIES = np.linspace(0.05, 1.5, 12)  # 濁り格子 [1/m]
N_SEEDS = 3
AZ_STARTS = 8                           # フォールバックの方位スタート数 (速度優先)

OPTICAL, FALLBACK = 0, 1

# 本シナリオは各 深度×濁り で**自動切替(§12)が達成する測位精度 (RMSE)** を色の濃淡で連続表示する
# 運用スペックマップ。§8 の現実誤差 (バイアス/音速ズレ/距離成長/外れ値/遅延) と §9 光減衰を反映する。
# 誤差を入れると単一距離フォールバックは脆くなり目標未達セルが増えるが、離散の運用可否(緑/金/赤)では
# なく達成RMSEの濃淡 (緑=高精度 → 赤=低精度) で示すので、どこでどれだけの精度が出るかが読み取れる。
# 目標精度の境界・切替境界は等高線で重ねる。幾何は config [survey] の near-nadir。
_ERR = dict(ERROR_MODEL) if ERROR_MODEL_ENABLE else {}   # §8 現実誤差を反映 (README参照)
_LOSS = "huber" if _ERR else "linear"     # 外れ値があるので robust で公平に推定


def _traj(depth):
    # 運用幾何: 子機は親機のほぼ真下 (near-nadir)。
    return double_lawnmower_trajectory(area=SURVEY_AREA, depth=-float(depth),
                                       n_legs=2, pts_per_leg=6, origin=SURVEY_ORIGIN)


def _opt_model(c):
    m = dict(OPTICAL_MODEL); m["attenuation_c"] = c
    return m


_REP_XY = (SURVEY_ORIGIN[0] + SURVEY_AREA[0] / 2.0,
           SURVEY_ORIGIN[1] + SURVEY_AREA[1] / 2.0)   # サーベイ中心 (代表水平位置)


def _range(depth):
    return float(np.linalg.norm([_REP_XY[0], _REP_XY[1], -depth]))   # 代表レンジ


def _optical_rmse(depth, clarity):
    """光学あり (減衰§9: 校正σ+見失い) + IMU + 深度, robust の軌道RMSE [mm] 平均。"""
    traj = _traj(depth)
    model = _opt_model(clarity)
    s_ang = optical_angular_sigma(float(np.linalg.norm(traj.mean(axis=0))), model)
    sig = (SIGMA[0], s_ang, s_ang)
    vals = []
    for s in range(N_SEEDS):
        ts = substream_seed(SEED, s)                        # 独立試行 (§15.2)
        z = simulate_observation_sequence_realistic(
            traj, SIGMA, seed=substream_seed(ts, 0), p_parent=P_PARENT,
            optical_model=model, **_ERR)
        z = apply_attitude_error_config(z, seed=substream_seed(ts, 4))  # §14 波動揺 (既定 OFF)
        imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=substream_seed(ts, 1))
        dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=substream_seed(ts, 2),
                                      bias=DEPTH_BIAS)
        est = estimate_trajectory(z, sig, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss=_LOSS,
                                  z_depth_seq=dep, sigma_depth=SIGMA_DEPTH)
        vals.append(rmse_xyz(traj, est)["total"] * 1000)
    return float(np.mean(vals))


def _fallback_rmse(depth):
    """距離+IMU+深度 (光学なし§11) の軌道RMSE [mm] 平均。濁り非依存。"""
    traj = _traj(depth)
    vals = []
    for s in range(N_SEEDS):
        ts = substream_seed(SEED, s)                        # 独立試行 (§15.2)
        if _ERR:           # 距離の現実誤差 (バイアス/音速ズレ/距離成長/遅延) を反映
            z = simulate_observation_sequence_realistic(
                traj, SIGMA, seed=substream_seed(ts, 0), p_parent=P_PARENT, **_ERR)
        else:
            z = simulate_observation_sequence(traj, SIGMA, seed=substream_seed(ts, 0),
                                              p_parent=P_PARENT)
        imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=substream_seed(ts, 1))
        dep = simulate_depth_sequence(traj, SIGMA_DEPTH, seed=substream_seed(ts, 2),
                                      bias=DEPTH_BIAS)
        est = estimate_trajectory_acoustic_inertial(
            z[:, 0], SIGMA[0], imu, SIGMA_IMU, dep, SIGMA_DEPTH,
            p_parent=P_PARENT, n_azimuth_starts=AZ_STARTS, loss=_LOSS)
        vals.append(rmse_xyz(traj, est)["total"] * 1000)
    return float(np.mean(vals))


def build_map():
    """各 (濁り×水深) で自動切替(§12)が達成する RMSE [mm] とモードを返す。

    達成精度 = そのセルで**実際に到達できる最良 RMSE**:
      - 見失い確率 p_drop <= しきい値 (光学が健全): 光学とフォールバックのうち高精度な方
      - p_drop > しきい値 (見失い多発): フォールバック (距離+IMU+深度, 濁り非依存) のみ
    mode はそのとき採用したモード。離散の運用可否ではなく、この達成RMSE を色の濃淡で見せる。
    """
    fb = {d: _fallback_rmse(d) for d in DEPTHS}     # フォールバックは濁り非依存 -> 水深ごと1回
    achieved = np.empty((len(CLARITIES), len(DEPTHS)))   # 達成RMSE [mm]
    mode = np.empty_like(achieved, dtype=int)            # OPTICAL / FALLBACK
    pdrop = np.empty_like(achieved)
    for i, c in enumerate(CLARITIES):
        model = _opt_model(c)
        for j, d in enumerate(DEPTHS):
            p = optical_dropout_prob(_range(d), model)
            pdrop[i, j] = p
            opt = _optical_rmse(d, c) if p <= P_SWITCH else None   # 光学が健全な時だけ評価
            if opt is not None and opt <= fb[d]:   # 光学が健全 かつ 光学が高精度
                achieved[i, j] = opt
                mode[i, j] = OPTICAL
            else:                            # フォールバックが良い / 見失い多発
                achieved[i, j] = fb[d]
                mode[i, j] = FALLBACK
    return achieved, mode, pdrop, fb


def main():
    print("=== 2次元運用スペック: 濁り×水深の達成精度マップ (§9-§12, 現実誤差込み) ===")
    print(f"フォント: {JP_FONT if USE_JP else '(英語ラベル)'} / 目標 {TARGET:.0f}mm "
          f"/ 切替しきい値 見失い率 {P_SWITCH:.2f} / 平均試行 {N_SEEDS}")

    achieved, mode, pdrop, fb = build_map()

    # コンソール要約: 各濁りでの 目標達成 最深水深 [m] と最小達成RMSE [mm]
    print("\n--- 各濁りでの達成精度 (自動切替) ---")
    print("  濁りc    目標達成 最深[m]   最小RMSE[mm]   その水深[m]")
    for i, c in enumerate(CLARITIES):
        ok_d = [DEPTHS[j] for j in range(len(DEPTHS)) if achieved[i, j] <= TARGET]
        max_d = max(ok_d) if ok_d else 0.0
        jbest = int(np.argmin(achieved[i, :]))
        print(f"  {c:5.2f}      {max_d:8.1f}        {achieved[i, jbest]:8.0f}     "
              f"{DEPTHS[jbest]:7.1f}")

    # ===== 達成RMSE ヒートマップ (色の濃淡で精度を連続表示) =====
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    dd = DEPTHS[1] - DEPTHS[0]
    cc = CLARITIES[1] - CLARITIES[0]
    extent = [DEPTHS[0] - dd / 2, DEPTHS[-1] + dd / 2,
              CLARITIES[0] - cc / 2, CLARITIES[-1] + cc / 2]
    vmin = max(1.0, float(np.nanmin(achieved)))
    vmax = float(np.nanmax(achieved))
    im = ax.imshow(achieved, origin="lower", aspect="auto", cmap="RdYlGn_r",
                   norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(Lbl("達成 RMSE [mm] (緑=高精度 / 赤=低精度)", "achieved RMSE [mm]"))

    X, Y = np.meshgrid(DEPTHS, CLARITIES)
    # 目標精度の境界 (達成RMSE = 目標): 左下側=目標達成。太い黒線。
    if vmin <= TARGET <= vmax:
        ct = ax.contour(X, Y, achieved, levels=[TARGET], colors="k", linewidths=2.2)
        ax.clabel(ct, fmt={TARGET: Lbl("目標%.0fmm" % TARGET, "target %.0fmm" % TARGET)},
                  fontsize=8)
    # 見失い率の切替境界 (p_drop = しきい値): 光学使用域とフォールバック域の境。白破線。
    cs = ax.contour(X, Y, pdrop, levels=[P_SWITCH], colors="white",
                    linewidths=1.6, linestyles="--")
    ax.clabel(cs, fmt={P_SWITCH: Lbl("切替境界(見失い%.0f%%)" % (P_SWITCH * 100),
                                     "switch %.0f%%" % (P_SWITCH * 100))}, fontsize=8)

    ax.set_xlabel(Lbl("水深 [m]", "depth [m]"))
    ax.set_ylabel(Lbl("濁り c [1/m] (clear→turbid)", "turbidity c [1/m]"))
    ax.set_title(Lbl(
        "運用スペックマップ: 到達できる最良測位精度 (色=RMSE, 緑=高精度→赤=低精度)\n"
        "太線=目標%.0fmm境界 / 白破線=光学↔フォールバック切替境界 (現実誤差§8+光減衰§9込み)" % TARGET,
        "Operational map: best achievable RMSE (green=good -> red=poor), target %.0fmm" % TARGET))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "operational_map.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # ===== 出力 =====
    mode_names = {OPTICAL: "optical", FALLBACK: "fallback"}
    payload = {
        "target_rmse_mm": TARGET, "switch_dropout_threshold": P_SWITCH,
        "error_model_enabled": bool(ERROR_MODEL_ENABLE),
        "depths_m": DEPTHS.tolist(), "clarities": CLARITIES.tolist(),
        "achieved_rmse_mm": np.round(achieved, 1).tolist(),
        "mode": mode.tolist(), "mode_names": mode_names,
        "fallback_rmse_mm": {("%.1f" % d): fb[d] for d in DEPTHS},
    }
    jpath = write_json("opmap/run_opmap", payload,
                       meta={"seed": int(SEED), "n_seeds": N_SEEDS,
                             "script": "run_opmap.py"})
    rows = []
    for i, c in enumerate(CLARITIES):
        for j, d in enumerate(DEPTHS):
            rows.append({"clarity_c": round(c, 3), "depth_m": round(d, 1),
                         "achieved_rmse_mm": round(achieved[i, j], 1),
                         "mode": mode_names[mode[i, j]],
                         "meets_target": bool(achieved[i, j] <= TARGET)})
    cpath = write_csv("opmap/run_opmap", rows,
                      header=["clarity_c", "depth_m", "achieved_rmse_mm",
                              "mode", "meets_target"])
    n_ok = int(np.count_nonzero(achieved <= TARGET))
    write_report(
        "opmap", "2次元運用スペック (濁り×水深の達成精度マップ)",
        "水深 × 濁り c の格子で、各条件で**到達できる最良の測位精度 (RMSE)** を色の濃淡で連続表示\n"
        "する。見失い率<=しきい値の区間は光学(角度+距離+IMU+深度)とフォールバックの高精度な方、\n"
        "超えた区間は距離+IMU+深度フォールバック(§11)。緑=高精度→赤=低精度。太線=目標境界、白破線=切替境界。\n"
        "現実誤差(§8)+光減衰(§9)を反映するので、目標未達のセルも『どれだけ達成できるか』が読み取れる。",
        condition_sections=["survey", "noise", "optical", "error_model", "acoustic",
                            "sync", "switch", "depth", "spec", "attitude"],
        not_reflected=[
            ("`[error_model] outlier_*` (フォールバック角度系)",
             "フォールバックは距離成分しか使わないので、方位/仰角の外れ値は作用しない "
             "(距離の外れ値・音速ズレ・距離成長・遅延は反映済み)。光学アームには全系統が作用する。"),
            ("`[attitude]`", "親機姿勢は固定と仮定 (波動揺は `run_attitude`)。"),
            ("`[sbl]` / `[stereo]`", "SBL・子機ステレオは使わない (別シナリオ)。"),
        ],
        outputs=[("operational_map.png", "達成RMSEの濃淡マップ (目標境界+切替境界つき)"),
                 ("run_opmap.json", "達成RMSE格子・モード・フォールバックRMSE"),
                 ("run_opmap.csv", "セルごとの達成RMSE/モード/目標達成可否")],
        results={"目標精度": f"{TARGET:.0f} mm",
                 "切替しきい値 (見失い率)": f"{P_SWITCH:.2f}",
                 "現実誤差(§8)": "反映" if ERROR_MODEL_ENABLE else "なし",
                 "目標達成セル数": f"{n_ok}/{achieved.size}",
                 "格子": f"水深{len(DEPTHS)} × 濁り{len(CLARITIES)}"},
        meta={"seed": SEED, "n_seeds": N_SEEDS}, math_spec="§9-§12")
    print(f"\n出力 : {FIGDIR}")
    print("\n完了。濁り×水深の達成RMSEを色の濃淡で表示 (現実誤差込み)。緑=高精度→赤=低精度、"
          "太線が目標境界。")


if __name__ == "__main__":
    main()
