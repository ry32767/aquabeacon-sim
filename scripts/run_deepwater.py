"""run_deepwater.py — 深い水深 (10〜20m) のテストシナリオ (MATH_SPEC §9)。

親機の光学追跡を「水中の減衰・拡散」モデル込みで評価する。深い/濁るほど:
  (1) SNR が落ち、角度ノイズ σ_ang が増える (測位精度が悪化)
  (2) ビーコン見失い (ドロップアウト=外れ値) が増える
  (3) その外れ値は IMU拘束つきロバスト推定 (§4.4) で抑えられる

水の濁り c [1/m] を clear/coastal/turbid と振り、水深を 5〜20m で掃引する。
最後に深い濁った水での軌道テストで linear vs robust を比較する。

出力: コンソール表 + figures/deepwater/deepwater.png + results/run_deepwater.{json,csv}
実行: python scripts/run_deepwater.py
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

from src.config import (SIGMA, SIGMA_IMU, P_PARENT, SEED, OPTICAL_MODEL,
                        DEEP_DEPTHS, DEEP_HORIZ_OFFSET, DEEP_CLARITIES,
                        DEEP_TRAJ_DEPTH, DEEP_TRAJ_CLARITY, DEEP_MC_N)
from src.truth import double_lawnmower_trajectory
from src.sensors import (optical_snr, optical_angular_sigma, optical_dropout_prob,
                         simulate_observation_realistic,
                         simulate_observation_sequence_realistic,
                         simulate_imu_displacements)
from src.estimator import estimate_position, estimate_trajectory
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_csv, scenario_dir, write_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = scenario_dir("deepwater")
_JP_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "Hiragino Sans", "TakaoPGothic", "IPAexGothic"]
_available = {f.name for f in fm.fontManager.ttflist}
_JP = next((c for c in _JP_CANDIDATES if c in _available), None)
USE_JP = _JP is not None
if USE_JP:
    plt.rcParams["font.family"] = _JP
plt.rcParams["axes.unicode_minus"] = False

CLARITY_LABEL = {0.05: "clear", 0.3: "coastal", 0.5: "coastal+", 1.0: "turbid"}


def Lbl(ja, en):
    return ja if USE_JP else en


def _model(c, dropout=True):
    """濁り c の光学モデルを作る (OPTICAL_MODEL をベースに c を差し替え)。"""
    m = dict(OPTICAL_MODEL)
    m["attenuation_c"] = c
    if not dropout:
        m["dropout_max"] = 0.0
    return m


def _truth_at_depth(depth):
    """水深 depth [m] の子機真位置 (水平オフセット込み)。z = -depth。"""
    return np.array([DEEP_HORIZ_OFFSET[0], DEEP_HORIZ_OFFSET[1], -float(depth)])


def _range_at_depth(depth):
    return float(np.linalg.norm(_truth_at_depth(depth)))


def _pos_rmse_mm(depth, model, seed=SEED, n=DEEP_MC_N):
    """光学モデル込みの測位 RMSE total [mm] (ドロップアウト含む)。"""
    truth = _truth_at_depth(depth)
    est = np.empty((n, 3))
    for i in range(n):
        z = simulate_observation_realistic(truth, SIGMA, seed=seed + i,
                                           p_parent=P_PARENT, optical_model=model)
        est[i] = estimate_position(z, SIGMA, p_parent=P_PARENT)
    return rmse_xyz(truth, est)["total"] * 1000


def sweep_curves():
    """濁りごとに、水深に対する σ_ang / SNR / dropout / 測位RMSE を返す。"""
    dense = np.linspace(min(DEEP_DEPTHS), max(DEEP_DEPTHS), 40)
    curves = {}
    for c in DEEP_CLARITIES:
        m = _model(c)
        rng = [_range_at_depth(d) for d in dense]
        curves[c] = {
            "depth_dense": dense.tolist(),
            "sigma_deg": [np.rad2deg(optical_angular_sigma(r, m)) for r in rng],
            "snr": [optical_snr(r, m) for r in rng],
            "dropout": [optical_dropout_prob(r, m) for r in rng],
        }
        # 測位RMSE はグリッド (粗) で MC。検出可能 (SNR>=snr_min) のみ。
        rmse = []
        for d in DEEP_DEPTHS:
            r = _range_at_depth(d)
            if optical_snr(r, m) >= m["snr_min"]:
                rmse.append(_pos_rmse_mm(d, m))
            else:
                rmse.append(None)            # 見失い支配域 (測位破綻)
        curves[c]["rmse_depths"] = list(DEEP_DEPTHS)
        curves[c]["rmse_mm"] = rmse
    return curves


def trajectory_demo(seed=SEED):
    """深い濁った水での軌道テスト: linear vs robust。"""
    depth, c = DEEP_TRAJ_DEPTH, DEEP_TRAJ_CLARITY
    traj = double_lawnmower_trajectory(area=(6.0, 4.0), depth=-depth,
                                       n_legs=2, pts_per_leg=6, origin=(3.0, 3.0))
    model = _model(c, dropout=True)
    z = simulate_observation_sequence_realistic(traj, SIGMA, seed=seed,
                                                p_parent=P_PARENT, optical_model=model)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 1)
    est_lin = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss="linear")
    est_rob = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT, loss="huber")
    r_lin = rmse_xyz(traj, est_lin)["total"] * 1000
    r_rob = rmse_xyz(traj, est_rob)["total"] * 1000
    p_drop = optical_dropout_prob(_range_at_depth(depth), model)
    return dict(traj=traj, est_lin=est_lin, est_rob=est_rob, depth=depth,
                clarity=c, p_drop=p_drop, rmse_lin=r_lin, rmse_rob=r_rob)


def _plot(curves, demo):
    fig = plt.figure(figsize=(17, 9))

    def _clarity_curves(ax, key, ylabel, title, logy=False, hline=None):
        for c in DEEP_CLARITIES:
            lab = "%s (c=%.2f)" % (CLARITY_LABEL.get(c, "c"), c)
            ax.plot(curves[c]["depth_dense"], curves[c][key], "-", label=lab)
        if hline is not None:
            ax.axhline(hline, color="k", ls="--", lw=0.8)
        ax.set_xlabel(Lbl("水深 [m]", "depth [m]"))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 1)
    _clarity_curves(ax, "sigma_deg", Lbl("角度ノイズ σ_ang [deg]", "sigma_ang [deg]"),
                    Lbl("(a) 角度精度の劣化", "(a) angular noise"), logy=True)
    ax = fig.add_subplot(2, 3, 2)
    _clarity_curves(ax, "snr", "SNR",
                    Lbl("(b) SNR (破線=検出しきい値)", "(b) SNR"), logy=True,
                    hline=OPTICAL_MODEL["snr_min"])
    ax = fig.add_subplot(2, 3, 3)
    _clarity_curves(ax, "dropout", Lbl("見失い確率", "dropout prob"),
                    Lbl("(c) ドロップアウト確率", "(c) dropout"))

    # (d) 測位RMSE vs 水深
    ax = fig.add_subplot(2, 3, 4)
    for c in DEEP_CLARITIES:
        ds = curves[c]["rmse_depths"]
        ys = curves[c]["rmse_mm"]
        xs = [d for d, y in zip(ds, ys) if y is not None]
        vy = [y for y in ys if y is not None]
        ax.plot(xs, vy, "o-", label="%s (c=%.2f)" % (CLARITY_LABEL.get(c, "c"), c))
    ax.set_xlabel(Lbl("水深 [m]", "depth [m]"))
    ax.set_ylabel("RMSE total [mm]")
    ax.set_title(Lbl("(d) 測位RMSE (検出可能域のみ)", "(d) positioning RMSE"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (e)(f) 軌道テスト linear vs robust (3D)
    traj = demo["traj"]
    for j, (key, title, col) in enumerate([
            ("est_lin", Lbl("(e) 純L2: 見失いに弱い", "(e) linear"), "gray"),
            ("est_rob", Lbl("(f) ロバスト(huber)", "(f) robust"), "tab:blue")]):
        ax = fig.add_subplot(2, 3, 5 + j, projection="3d")
        est = demo[key]
        rmse = demo["rmse_lin"] if j == 0 else demo["rmse_rob"]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.8,
                label=Lbl("真の軌道", "true"))
        ax.scatter(est[:, 0], est[:, 1], est[:, 2], c=col, s=22,
                   label=Lbl("推定", "estimate"))
        ax.set_title("%s  RMSE %.0f mm" % (title, rmse), fontsize=10)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.legend(fontsize=8, loc="upper left")
        ax.view_init(elev=38, azim=-65)

    fig.suptitle(Lbl(
        "深い水深テスト: 水中の光減衰で精度劣化・見失い増 → ロバスト推定で軌道を救う "
        "(深さ%.0fm, 濁りc=%.1f, 見失い~%.0f%%)" % (
            demo["depth"], demo["clarity"], demo["p_drop"] * 100),
        "Deep-water test: optical attenuation degrades tracking; robust estimation recovers"))
    fig.tight_layout()
    png = os.path.join(FIGDIR, "deepwater.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def main(seed=SEED):
    print("=== 深い水深テストシナリオ (MATH_SPEC §9) ===")
    print(f"フォント: {_JP if USE_JP else '(英語ラベル)'} / "
          f"水平オフセット={DEEP_HORIZ_OFFSET.tolist()} m / MC試行={DEEP_MC_N}")

    curves = sweep_curves()
    print("\n--- 測位スペック (光学減衰込み): RMSE total [mm] / '---'=見失い支配で破綻 ---")
    header = "  濁りc      " + "".join("%7sm" % d for d in DEEP_DEPTHS)
    print(header)
    for c in DEEP_CLARITIES:
        cells = []
        for y in curves[c]["rmse_mm"]:
            cells.append("   ---  " if y is None else "%7.0f " % y)
        print(f"  {CLARITY_LABEL.get(c,'c'):8s}({c:.2f})" + "".join(cells))

    demo = trajectory_demo(seed=seed)
    print(f"\n--- 軌道テスト (深さ{demo['depth']:.0f}m, 濁りc={demo['clarity']:.1f}, "
          f"見失い~{demo['p_drop']*100:.0f}%) ---")
    print(f"  純L2 (linear)   RMSE = {demo['rmse_lin']:6.0f} mm")
    print(f"  ロバスト(huber) RMSE = {demo['rmse_rob']:6.0f} mm")
    imp = (1 - demo["rmse_rob"] / demo["rmse_lin"]) * 100 if demo["rmse_lin"] > 0 else 0
    print(f"  -> ロバストで {imp:.0f}% 改善")

    png = _plot(curves, demo)

    payload = {
        "horiz_offset_m": DEEP_HORIZ_OFFSET.tolist(),
        "positioning_rmse": {
            CLARITY_LABEL.get(c, "c%.2f" % c): {
                "clarity_c": c,
                "depths_m": curves[c]["rmse_depths"],
                "rmse_mm": curves[c]["rmse_mm"],
            } for c in DEEP_CLARITIES},
        "trajectory_demo": {
            "depth_m": demo["depth"], "clarity_c": demo["clarity"],
            "expected_dropout_pct": demo["p_drop"] * 100,
            "rmse_linear_mm": demo["rmse_lin"], "rmse_robust_mm": demo["rmse_rob"]},
    }
    jpath = write_json("deepwater/run_deepwater", payload,
                       meta={"seed": int(seed), "mc_n": int(DEEP_MC_N),
                             "script": "run_deepwater.py"})
    csv_rows = []
    for c in DEEP_CLARITIES:
        for d, y in zip(curves[c]["rmse_depths"], curves[c]["rmse_mm"]):
            csv_rows.append({"clarity_c": c, "depth_m": d,
                             "rmse_mm": ("" if y is None else round(y, 1))})
    cpath = write_csv("deepwater/run_deepwater", csv_rows,
                      header=["clarity_c", "depth_m", "rmse_mm"])
    write_report(
        "deepwater", "深い水深テスト (光学減衰・見失い)",
        "親機の光学追跡を水中の減衰・拡散モデル込みで評価する。水の濁り c を clear/coastal/turbid と\n"
        "振り、水深 5〜20m で測位精度 (RMSE) と見失い (ドロップアウト) を掃引する。深い/濁るほど\n"
        "SNR が落ちて角度精度が悪化し見失いが増える。最後に深い濁り水の軌道を linear vs robust で比較する。",
        condition_sections=["noise", "optical", "deepwater", "estimator"],
        outputs=[("deepwater.png", "σ_ang/SNR/見失い/測位RMSE/linear vs robust軌道"),
                 ("run_deepwater.json", "曲線データと軌道比較"),
                 ("run_deepwater.csv", "濁り×水深の測位RMSE")],
        results={"軌道デモ条件": f"深さ{demo['depth']:.0f}m, 濁りc={demo['clarity']:.1f}, "
                 f"見失い~{demo['p_drop']*100:.0f}%",
                 "linear RMSE": f"{demo['rmse_lin']:.0f} mm",
                 "robust(huber) RMSE": f"{demo['rmse_rob']:.0f} mm"},
        meta={"seed": seed, "mc_n": DEEP_MC_N}, math_spec="§9")
    print(f"\n出力 : {FIGDIR}")
    print("\n完了。深い水深で光減衰が測位を悪化させ、ロバスト推定が見失いを救うことを確認。")


if __name__ == "__main__":
    main()
