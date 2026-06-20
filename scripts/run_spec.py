"""run_spec.py — 設計スペックシート: 感度掃引を「設計要求」に逆算する。

MBD の最大の成果物。「目標精度を満たすには各設計パラメータをどこに置くべきか」を
数値で出す。例:
  - 測位 RMSE ≤ 100 mm を満たす最大距離 d / 最大角度ノイズ σ_ang
  - マッピング寸法誤差 ≤ 30 mm を満たす最大 standoff / 最小ベースライン /
    最大カメラ角度ノイズ σ_cam / 最小フレーム数 looks

各パラメータを1つずつ掃引して指標を計算し、目標線を横切る境界を線形補間で求める。
出力:
  - コンソールに要求仕様の表
  - results/run_spec.{json,csv}  (機械可読)
  - figures/spec/design_spec.png (目標線つきグラフ)

config.toml [spec] で目標値・探索グリッドを編集できる。
config.toml [error_model] enable=true なら、測位スペックは現実的誤差込みで評価する
(系統バイアス等が要求を厳しくする様子が出る)。

実行: python scripts/run_spec.py
MBD: 推定/三角測量には truth を渡さない。評価でだけ真値と突き合わせる。
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_DIST, P_PARENT, CUBE_SIDE, CUBE_CENTER,
                        STEREO_BASELINE, STEREO_SIGMA_CAM, STEREO_STANDOFF,
                        ERROR_MODEL, ERROR_MODEL_ENABLE,
                        SPEC_POS_RMSE_TARGET_MM, SPEC_POS_BEARING,
                        SPEC_POS_RANGE_GRID, SPEC_POS_NOMINAL_RANGE,
                        SPEC_POS_SIGMA_ANG_GRID, SPEC_GEOM_DIM_TARGET_MM,
                        SPEC_GEOM_STANDOFF_GRID, SPEC_GEOM_BASELINE_GRID,
                        SPEC_GEOM_SIGMA_CAM_GRID, SPEC_GEOM_LOOKS_GRID,
                        SPEC_GEOM_N_PER_EDGE, SPEC_MC_N, SEED)
from src.truth import true_cube_pointcloud
from src.sensors import (simulate_observation_realistic, stereo_camera_positions,
                         simulate_stereo_observation)
from src.estimator import estimate_position
from src.geometry import stereo_triangulate, robust_cube_side_estimate
from src.evaluation import (monte_carlo_rmse, rmse_xyz, dimension_error_mm)
from src.results_io import write_json, write_csv

# --- 出力先・日本語フォント ---------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures", "spec")
os.makedirs(FIGDIR, exist_ok=True)
_JP_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "Hiragino Sans", "TakaoPGothic", "IPAexGothic"]
_available = {f.name for f in fm.fontManager.ttflist}
_JP = next((c for c in _JP_CANDIDATES if c in _available), None)
USE_JP = _JP is not None
if USE_JP:
    plt.rcParams["font.family"] = _JP
plt.rcParams["axes.unicode_minus"] = False


def Lbl(ja, en):
    return ja if USE_JP else en


# ----------------------------------------------------------------------------
# 境界探索: ys が target を横切る x を線形補間で求める
# ----------------------------------------------------------------------------
def _interp_crossing(xs, ys, target):
    """単調な (xs, ys) で ys=target となる x を線形補間。無ければ None。"""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            t = (target - y0) / (y1 - y0)
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def _requirement(param, unit, xs, ys, target, sense):
    """掃引結果から設計要求を組み立てる。

    sense='le' : 小さいほど良い (指標は x とともに悪化) -> 要求 "x <= x*"
    sense='ge' : 大きいほど良い (指標は x とともに改善) -> 要求 "x >= x*"
    戻り値: 表示・保存用の dict。
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    x_star = _interp_crossing(xs, ys, target)
    ok = ys <= target
    if x_star is not None:
        boundary = x_star
        if sense == "le":
            req = f"{param} <= {x_star:.3g} {unit}"
        else:
            req = f"{param} >= {x_star:.3g} {unit}"
        achievable = True
    else:
        boundary = None
        if ok.all():
            # 全グリッドで達成。境界はグリッド端 (より厳しい側) を提示。
            edge = xs.max() if sense == "le" else xs.min()
            req = (f"{param} <= {edge:.3g} {unit} (掃引範囲すべて達成。さらに広げて確認可)"
                   if sense == "le" else
                   f"{param} >= {edge:.3g} {unit} (掃引範囲すべて達成。さらに広げて確認可)")
            achievable = True
        else:
            req = f"掃引範囲内に達成点なし (目標 {target:g} に届かず)"
            achievable = False
    return {
        "parameter": param,
        "unit": unit,
        "target_metric_value": float(target),
        "boundary": (None if boundary is None else float(boundary)),
        "requirement": req,
        "achievable": bool(achievable),
        "sweep_x": [float(v) for v in xs],
        "sweep_metric": [float(v) for v in ys],
    }


# ----------------------------------------------------------------------------
# 指標の計算
# ----------------------------------------------------------------------------
def _pos_rmse_mm(truth, sigma, seed=SEED, n=SPEC_MC_N):
    """測位 RMSE total [mm]。error_model 有効なら現実的誤差込みで評価する。"""
    if ERROR_MODEL_ENABLE:
        est = np.empty((n, 3))
        for i in range(n):
            z = simulate_observation_realistic(truth, sigma, seed=seed + i,
                                               p_parent=P_PARENT, **ERROR_MODEL)
            est[i] = estimate_position(z, sigma, p_parent=P_PARENT)
        return rmse_xyz(truth, est)["total"] * 1000
    return monte_carlo_rmse(truth, sigma, n=n, seed=seed,
                            p_parent=P_PARENT)["total"] * 1000


def _truth_at_range(d):
    """指定距離 d [m] にある真値位置 (掃引方向 SPEC_POS_BEARING を正規化して使う)。"""
    u = SPEC_POS_BEARING / np.linalg.norm(SPEC_POS_BEARING)
    return P_PARENT + d * u


def _stereo_dim_error_mm(cloud, standoff, baseline, sigma_cam, looks, seed=SEED):
    """子機ステレオでキューブを復元したときのロバスト寸法誤差 |L誤差| [mm]。

    三角測量には truth を渡さない (観測の方位 + 既知カメラ位置のみ)。
    """
    est = np.empty_like(cloud)
    for i, p in enumerate(cloud):
        c_L, c_R = stereo_camera_positions(p, CUBE_CENTER, standoff, baseline)
        acc = np.zeros(3)
        for m in range(looks):
            brg = simulate_stereo_observation(p, c_L, c_R, sigma_cam,
                                              seed=seed + i * 1000 + m)
            acc += stereo_triangulate(brg, c_L, c_R)
        est[i] = acc / looks
    return abs(dimension_error_mm(robust_cube_side_estimate(est), CUBE_SIDE))


# ----------------------------------------------------------------------------
# 掃引 (測位 / ジオメトリ)
# ----------------------------------------------------------------------------
def sweep_positioning():
    target = SPEC_POS_RMSE_TARGET_MM
    rows = []

    # (1) 距離 d を掃引: 遠いほど RMSE 悪化 -> "d <= d*"
    ds = SPEC_POS_RANGE_GRID
    rmse_d = [_pos_rmse_mm(_truth_at_range(d), SIGMA) for d in ds]
    rows.append(_requirement(Lbl("距離 d", "range d"), "m", ds, rmse_d, target, "le"))

    # (2) 角度ノイズ σ_ang を掃引 (距離は nominal 固定): 大きいほど悪化 -> "σ <= σ*"
    truth = _truth_at_range(SPEC_POS_NOMINAL_RANGE)
    sigs = SPEC_POS_SIGMA_ANG_GRID
    rmse_a = []
    for deg in sigs:
        sig = (SIGMA_DIST, np.deg2rad(deg), np.deg2rad(deg))
        rmse_a.append(_pos_rmse_mm(truth, sig))
    rows.append(_requirement(Lbl("角度ノイズ σ_ang", "sigma_ang"), "deg",
                             sigs, rmse_a, target, "le"))
    return target, rows


def sweep_geometry():
    target = SPEC_GEOM_DIM_TARGET_MM
    cloud = true_cube_pointcloud(n_per_edge=SPEC_GEOM_N_PER_EDGE)
    looks_fixed = 15
    rows = []

    # (1) standoff: 遠いほど悪化 (奥行き誤差 ∝ Z^2) -> "standoff <= S*"
    Ss = SPEC_GEOM_STANDOFF_GRID
    de_s = [_stereo_dim_error_mm(cloud, S, STEREO_BASELINE, STEREO_SIGMA_CAM,
                                 looks_fixed) for S in Ss]
    rows.append(_requirement(Lbl("観測距離 standoff", "standoff"), "m",
                             Ss, de_s, target, "le"))

    # (2) ベースライン B: 長いほど改善 -> "B >= B*"
    Bs = SPEC_GEOM_BASELINE_GRID
    de_b = [_stereo_dim_error_mm(cloud, STEREO_STANDOFF, B, STEREO_SIGMA_CAM,
                                 looks_fixed) for B in Bs]
    rows.append(_requirement(Lbl("ベースライン B", "baseline B"), "m",
                             Bs, de_b, target, "ge"))

    # (3) カメラ角度ノイズ σ_cam: 小さいほど良い -> "σ_cam <= σ*"
    cams = SPEC_GEOM_SIGMA_CAM_GRID
    de_c = [_stereo_dim_error_mm(cloud, STEREO_STANDOFF, STEREO_BASELINE,
                                 np.deg2rad(deg), looks_fixed) for deg in cams]
    rows.append(_requirement(Lbl("カメラ角度ノイズ σ_cam", "sigma_cam"), "deg",
                             cams, de_c, target, "le"))

    # (4) フレーム数 looks: 多いほど改善 -> "looks >= N*"
    Ls = SPEC_GEOM_LOOKS_GRID
    de_l = [_stereo_dim_error_mm(cloud, STEREO_STANDOFF, STEREO_BASELINE,
                                 STEREO_SIGMA_CAM, int(k)) for k in Ls]
    rows.append(_requirement(Lbl("フレーム数 looks", "frames looks"), "",
                             Ls, de_l, target, "ge"))
    return target, rows


# ----------------------------------------------------------------------------
# 出力 (コンソール表 + 図)
# ----------------------------------------------------------------------------
def _print_table(title, target, unit, rows):
    print(f"\n=== {title}  (目標: 指標 <= {target:g} {unit}) ===")
    for r in rows:
        mark = "OK " if r["achievable"] else "NG "
        print(f"  [{mark}] {r['requirement']}")


def _plot(target_pos, pos_rows, target_geom, geom_rows):
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    panels = []
    # 測位2枚
    panels.append((axs[0, 0], pos_rows[0], target_pos, "RMSE [mm]",
                   Lbl("(測位a) 距離 vs RMSE", "(pos a) range vs RMSE")))
    panels.append((axs[0, 1], pos_rows[1], target_pos, "RMSE [mm]",
                   Lbl("(測位b) 角度ノイズ vs RMSE", "(pos b) sigma_ang vs RMSE")))
    # ジオメトリ4枚
    titles = [Lbl("(幾何a) standoff vs 寸法誤差", "(geo a) standoff"),
              Lbl("(幾何b) ベースライン vs 寸法誤差", "(geo b) baseline"),
              Lbl("(幾何c) σ_cam vs 寸法誤差", "(geo c) sigma_cam"),
              Lbl("(幾何d) looks vs 寸法誤差", "(geo d) looks")]
    slots = [axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]
    for ax, r, t in zip(slots, geom_rows, titles):
        panels.append((ax, r, target_geom, Lbl("寸法誤差 |L誤差| [mm]", "|dim err| [mm]"), t))

    for ax, r, target, ylab, title in panels:
        xs, ys = r["sweep_x"], r["sweep_metric"]
        ax.plot(xs, ys, "o-", color="tab:blue", label=Lbl("指標", "metric"))
        ax.axhline(target, color="tab:red", ls="--",
                   label=Lbl("目標 %g" % target, "target %g" % target))
        if r["boundary"] is not None:
            ax.axvline(r["boundary"], color="tab:green", ls=":",
                       label=Lbl("要求境界", "boundary"))
        ax.set_xlabel(f"{r['parameter']} [{r['unit']}]" if r['unit'] else r['parameter'])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    mode = Lbl("現実誤差込み", "with realistic errors") if ERROR_MODEL_ENABLE \
        else Lbl("理想ノイズ", "ideal noise")
    fig.suptitle(Lbl(
        "設計スペックシート: 目標精度を満たす設計要求 (測位=%s / 幾何=理想ステレオ)" % mode,
        "Design spec sheet: requirements to meet target accuracy"))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "design_spec.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    print("=== 設計スペックシート (run_spec.py) ===")
    print(f"フォント: {_JP if USE_JP else '(英語ラベル)'} / "
          f"測位評価: {'現実誤差込み' if ERROR_MODEL_ENABLE else '理想ノイズ'}")

    target_pos, pos_rows = sweep_positioning()
    _print_table(Lbl("測位スペック (親機カメラ+音響)", "Positioning spec"),
                 target_pos, "mm", pos_rows)

    target_geom, geom_rows = sweep_geometry()
    _print_table(Lbl("ジオメトリ スペック (子機ステレオ)", "Geometry spec"),
                 target_geom, "mm", geom_rows)

    png = _plot(target_pos, pos_rows, target_geom, geom_rows)

    # --- 機械可読出力 ---
    payload = {
        "positioning": {"target_rmse_mm": target_pos, "requirements": pos_rows},
        "geometry": {"target_dim_error_mm": target_geom, "requirements": geom_rows},
    }
    meta = {"seed": int(SEED), "mc_n": int(SPEC_MC_N),
            "error_model_enabled": bool(ERROR_MODEL_ENABLE), "script": "run_spec.py"}
    jpath = write_json("run_spec", payload, meta=meta)
    csv_rows = []
    for sub, rs in (("positioning", pos_rows), ("geometry", geom_rows)):
        for r in rs:
            csv_rows.append({"subsystem": sub, "parameter": r["parameter"],
                             "unit": r["unit"], "requirement": r["requirement"],
                             "boundary": r["boundary"], "achievable": r["achievable"]})
    cpath = write_csv("run_spec", csv_rows,
                      header=["subsystem", "parameter", "unit", "requirement",
                              "boundary", "achievable"])
    print(f"\n図   : {png}")
    print(f"JSON : {jpath}")
    print(f"CSV  : {cpath}")
    print("\n完了。目標精度を満たす設計要求 (距離/角度/standoff/baseline/σ_cam/looks) を出力。")


if __name__ == "__main__":
    main()
