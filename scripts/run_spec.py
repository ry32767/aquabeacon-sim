"""run_spec.py — 設計スペックシート: 感度掃引を「設計要求」に逆算する。

MBD の最大の成果物。「目標精度を満たすには各設計パラメータをどこに置くべきか」を
数値で出す。例:
  - 測位 RMSE ≤ 100 mm を満たす最大距離 d / 最大角度ノイズ σ_ang
  - マッピング寸法誤差 ≤ 30 mm を満たす最大 standoff / 最小ベースライン /
    最大カメラ角度ノイズ σ_cam / 最小フレーム数 looks
  - 運用可能な最大水深 (光学減衰§9 + 深度センサ§10): 濁りごとに、深度センサ
    あり/なしで「目標 RMSE を満たす最大水深」を求める (深度センサが延伸する)

各パラメータを1つずつ掃引して指標を計算し、目標線を横切る境界を線形補間で求める。
出力:
  - コンソールに要求仕様の表 + 最大運用水深の表
  - results/run_spec.{json,csv}  (機械可読)
  - figures/spec/design_spec.png (設計要求グラフ)
  - figures/spec/operational_depth.png (最大運用水深: 深度センサあり/なし)

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
                        SPEC_GEOM_N_PER_EDGE, SPEC_MC_N, SEED, SPEC_OPDEPTH_TARGET_MM,
                        OPTICAL_MODEL, SIGMA_DEPTH, DEEP_CLARITIES, DEEP_HORIZ_OFFSET)
from src.truth import true_cube_pointcloud
from src.sensors import (simulate_observation, simulate_observation_realistic,
                         simulate_depth, stereo_camera_positions,
                         simulate_stereo_observation,
                         optical_angular_sigma, optical_snr)
from src.estimator import estimate_position
from src.geometry import stereo_triangulate, robust_cube_side_estimate
from src.evaluation import (monte_carlo_rmse, rmse_xyz, dimension_error_mm)
from src.results_io import write_json, write_csv

CLARITY_LABEL = {0.05: "clear", 0.3: "coastal", 0.5: "coastal+", 1.0: "turbid"}

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
# 運用可能な最大水深 (光学減衰 §9 + 深度センサ §10 の統合)
#
# 深い/濁った水では光減衰で角度 σ_ang が増え (§9)、測位 RMSE が悪化する。
# 深度センサ (§10) は鉛直 z を直接拘束して RMSE を下げ、目標を満たす最大水深を伸ばす。
# ただし SNR < snr_min で光ビーコンを見失うと測位自体が成立しない (検出限界)。
# 角度の重みは σ_ang(d) を採用する well-calibrated 推定 (適応重み) を仮定する。
# ----------------------------------------------------------------------------
OPDEPTH_N = min(SPEC_MC_N, 500)         # 運用水深掃引の MC 試行数 (速度優先)
OPDEPTH_SCAN = np.arange(3.0, 25.0 + 0.01, 1.5)   # 水深スキャン [m]


def _opt_model(c):
    m = dict(OPTICAL_MODEL)
    m["attenuation_c"] = c
    return m


def _truth_depth(depth):
    return np.array([DEEP_HORIZ_OFFSET[0], DEEP_HORIZ_OFFSET[1], -float(depth)])


def _range_depth(depth):
    return float(np.linalg.norm(_truth_depth(depth)))


def _sigma_calibrated(depth, model):
    """その水深の光学 σ_ang(d) を反映した観測 σ (適応重み)。"""
    s = optical_angular_sigma(_range_depth(depth), model)
    return (SIGMA_DIST, s, s)


def _pos_rmse_depth(depth, model, use_depth, seed=SEED, n=OPDEPTH_N):
    """光学減衰込み + 深度センサあり/なし の測位 RMSE total [mm]。"""
    truth = _truth_depth(depth)
    sig = _sigma_calibrated(depth, model)
    est = np.empty((n, 3))
    for i in range(n):
        z = simulate_observation(truth, sig, seed=seed + i, p_parent=P_PARENT)
        if use_depth:
            zd = simulate_depth(truth, SIGMA_DEPTH, seed=seed + 700000 + i)
            est[i] = estimate_position(z, sig, p_parent=P_PARENT,
                                       z_depth=zd, sigma_depth=SIGMA_DEPTH)
        else:
            est[i] = estimate_position(z, sig, p_parent=P_PARENT)
    return rmse_xyz(truth, est)["total"] * 1000


def _op_depth(depths, rmse, target, det_limit):
    """最大運用水深 = min(RMSEが目標を超える水深, 検出限界)。制約名も返す。

    binding: 'precision' (RMSEが縛る/深度センサで伸ばせる) / 'detection' (見失いが縛る) /
             'none' (最浅でも目標未達 = その濁りでは運用不可)。
    """
    if len(depths) == 0:
        return 0.0, "detection"
    depths = list(depths)
    if rmse[0] > target:                       # 最浅でも未達
        return 0.0, "none"
    cross = _interp_crossing(depths, rmse, target)
    if cross is not None:
        if cross <= det_limit:
            return float(cross), "precision"
        return float(det_limit), "detection"
    return float(det_limit), "detection"       # 全検出域で達成 -> 検出が縛る


def sweep_operational_depth():
    """濁りごとに、深度センサあり/なしの最大運用水深を求める (§9 + §10)。"""
    target = SPEC_OPDEPTH_TARGET_MM            # ミッション精度 (best-case より緩い)
    snr_min = OPTICAL_MODEL["snr_min"]
    results = []
    for c in DEEP_CLARITIES:
        m = _opt_model(c)
        snr_scan = [optical_snr(_range_depth(d), m) for d in OPDEPTH_SCAN]
        det = [d for d, s in zip(OPDEPTH_SCAN, snr_scan) if s >= snr_min]
        det_limit = float(max(det)) if det else float(OPDEPTH_SCAN[0])
        dd = np.array([d for d in OPDEPTH_SCAN if d <= det_limit])
        rmse_no = [_pos_rmse_depth(d, m, False) for d in dd]
        rmse_dp = [_pos_rmse_depth(d, m, True) for d in dd]
        max_no, bind_no = _op_depth(dd, rmse_no, target, det_limit)
        max_dp, bind_dp = _op_depth(dd, rmse_dp, target, det_limit)
        results.append(dict(
            clarity=c, label=CLARITY_LABEL.get(c, "c=%.2f" % c), det_limit=det_limit,
            depths=dd.tolist(), rmse_no=rmse_no, rmse_dp=rmse_dp,
            scan=OPDEPTH_SCAN.tolist(), snr_scan=snr_scan,
            max_no=max_no, bind_no=bind_no, max_dp=max_dp, bind_dp=bind_dp,
            gain=max_dp - max_no))
    return target, results


# ----------------------------------------------------------------------------
# 出力 (コンソール表 + 図)
# ----------------------------------------------------------------------------
def _print_table(title, target, unit, rows):
    print(f"\n=== {title}  (目標: 指標 <= {target:g} {unit}) ===")
    for r in rows:
        mark = "OK " if r["achievable"] else "NG "
        print(f"  [{mark}] {r['requirement']}")


_BIND_JA = {"precision": "精度律速", "detection": "検出律速", "none": "運用不可"}


def _print_operational_depth(target, results):
    print(f"\n=== 運用可能な最大水深 (光学減衰§9 + 深度センサ§10)  "
          f"(目標: 測位RMSE <= {target:g} mm) ===")
    print("  濁り          深度なし   深度あり    伸び    制約(あり)  検出限界")
    for r in results:
        print("  %-9s(%.2f) %7.1fm %8.1fm %7.1fm   %-8s %7.1fm" % (
            r["label"], r["clarity"], r["max_no"], r["max_dp"], r["gain"],
            _BIND_JA.get(r["bind_dp"], r["bind_dp"]), r["det_limit"]))


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


def _plot_operational_depth(target, results):
    """濁りごとの RMSE vs 水深 (深度なし/あり) + 最大運用水深の要約棒グラフ。"""
    n = len(results)
    fig, axs = plt.subplots(1, n + 1, figsize=(5.0 * (n + 1), 4.6))

    for ax, r in zip(axs[:n], results):
        ax.plot(r["depths"], r["rmse_no"], "o-", color="gray",
                label=Lbl("深度なし", "no depth"))
        ax.plot(r["depths"], r["rmse_dp"], "o-", color="tab:blue",
                label=Lbl("深度あり", "with depth"))
        ax.axhline(target, color="tab:red", ls="--", label=Lbl("目標", "target"))
        ax.axvline(r["det_limit"], color="k", ls=":", alpha=0.6,
                   label=Lbl("検出限界", "detection limit"))
        for mx, col in [(r["max_no"], "gray"), (r["max_dp"], "tab:blue")]:
            if mx > 0:
                ax.axvline(mx, color=col, ls="-", alpha=0.35)
        ax.set_xlabel(Lbl("水深 [m]", "depth [m]"))
        ax.set_ylabel("RMSE total [mm]")
        ax.set_title("%s (c=%.2f)" % (r["label"], r["clarity"]))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # 要約棒グラフ: 最大運用水深 (深度なし vs あり)
    axb = axs[n]
    x = np.arange(n)
    axb.bar(x - 0.2, [r["max_no"] for r in results], 0.4, color="gray",
            label=Lbl("深度なし", "no depth"))
    axb.bar(x + 0.2, [r["max_dp"] for r in results], 0.4, color="tab:blue",
            label=Lbl("深度あり", "with depth"))
    axb.set_xticks(x)
    axb.set_xticklabels(["%s\n(c=%.2f)" % (r["label"], r["clarity"]) for r in results],
                        fontsize=8)
    axb.set_ylabel(Lbl("最大運用水深 [m]", "max operational depth [m]"))
    axb.set_title(Lbl("最大運用水深: 深度センサで延伸", "max operational depth"))
    axb.grid(alpha=0.3, axis="y")
    axb.legend(fontsize=8)

    fig.suptitle(Lbl(
        "運用可能な最大水深: 光学減衰(§9)で深いほど精度劣化、深度センサ(§10)で延伸 "
        "(目標 RMSE <= %g mm)" % target,
        "Max operational depth: depth sensor extends it (target %g mm)" % target))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "operational_depth.png")
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

    target_op, op_results = sweep_operational_depth()
    _print_operational_depth(target_op, op_results)

    png = _plot(target_pos, pos_rows, target_geom, geom_rows)
    png_op = _plot_operational_depth(target_op, op_results)

    # --- 機械可読出力 ---
    op_payload = [{
        "clarity_c": r["clarity"], "label": r["label"],
        "max_op_depth_no_depth_m": r["max_no"], "max_op_depth_with_depth_m": r["max_dp"],
        "gain_m": r["gain"], "binding_with_depth": r["bind_dp"],
        "detection_limit_m": r["det_limit"],
    } for r in op_results]
    payload = {
        "positioning": {"target_rmse_mm": target_pos, "requirements": pos_rows},
        "geometry": {"target_dim_error_mm": target_geom, "requirements": geom_rows},
        "operational_depth": {"target_rmse_mm": target_op, "by_clarity": op_payload},
    }
    meta = {"seed": int(SEED), "mc_n": int(SPEC_MC_N),
            "error_model_enabled": bool(ERROR_MODEL_ENABLE),
            "sigma_depth_m": SIGMA_DEPTH, "script": "run_spec.py"}
    jpath = write_json("run_spec", payload, meta=meta)
    csv_rows = []
    for sub, rs in (("positioning", pos_rows), ("geometry", geom_rows)):
        for r in rs:
            csv_rows.append({"subsystem": sub, "parameter": r["parameter"],
                             "unit": r["unit"], "requirement": r["requirement"],
                             "boundary": r["boundary"], "achievable": r["achievable"]})
    for r in op_results:
        csv_rows.append({"subsystem": "operational_depth",
                         "parameter": "%s (c=%.2f)" % (r["label"], r["clarity"]),
                         "unit": "m",
                         "requirement": "最大運用水深 深度なし %.1fm / 深度あり %.1fm (%s)"
                         % (r["max_no"], r["max_dp"], _BIND_JA.get(r["bind_dp"])),
                         "boundary": r["max_dp"], "achievable": (r["max_dp"] > 0)})
    cpath = write_csv("run_spec", csv_rows,
                      header=["subsystem", "parameter", "unit", "requirement",
                              "boundary", "achievable"])
    print(f"\n図   : {png}")
    print(f"図   : {png_op}")
    print(f"JSON : {jpath}")
    print(f"CSV  : {cpath}")
    print("\n完了。設計要求 (距離/角度/standoff/baseline/σ_cam/looks) と "
          "深度センサあり/なしの最大運用水深を出力。")


if __name__ == "__main__":
    main()
