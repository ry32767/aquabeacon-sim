"""
run_mapping.py — Stage 2 (サクセス: マッピング) を1本通すスクリプト。

2 つのことを示す:
  (A) 複数時刻の軌道推定 (MATH_SPEC §5): ダブル芝刈り軌道を、観測のみ / 観測+IMU で
      推定し、RMSE を比較する (IMU 拘束で精度が上がることを確認)。
  (B) 既知キューブのジオメトリ評価 (MATH_SPEC §6, §6.2): キューブ表面の各点を
      子機の2カメラ(ステレオ)で観測し、三角測量で推定点群を作り、寸法 L_hat /
      体積 V_hat と誤差を出す。位置推定(親機カメラ+音響)とは別系統。

フロー (MBD):
  ① truth     : 真の軌道・真のキューブ表面
  ② sensors   : (測位) ノイズ付き観測列 + IMU 変位 / (ジオメトリ) ステレオ2方位
  ③ estimator : 観測のみから軌道を推定 (truth は渡さない)
  ④ geometry  : ステレオ三角測量 -> 点群 -> 寸法・体積 (truth を見ない)
  ⑤ evaluation: 真値 vs 推定で RMSE・寸法誤差・体積誤差率

乱数は seed 固定で再現可能。
"""
import os
import sys

import numpy as np

from _plotstyle import plt, USE_JP, Lbl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_IMU, P_PARENT, CUBE_SIDE, CUBE_CENTER,
                        CUBE_N_PER_EDGE, MAP_LOOKS, SEED,
                        STEREO_BASELINE, STEREO_SIGMA_CAM, STEREO_STANDOFF, STEREO_UP)
from src.truth import double_lawnmower_trajectory, true_cube_pointcloud
from src.sensors import (simulate_observation_sequence, simulate_imu_displacements,
                         stereo_camera_positions, simulate_stereo_observation,
                         apply_attitude_error_config)
from src.estimator import estimate_trajectory
from src.geometry import (aabb_volume, cube_side_estimate, robust_volume,
                          robust_cube_side_estimate, stereo_triangulate)
from src.evaluation import (rmse_xyz, dimension_error_mm, volume_error_rate_pct,
                            pointcloud_rms_to_surface)
from src.results_io import write_json, write_report, scenario_dir

FIGDIR = scenario_dir("mapping")


def _plot_trajectory(traj, est_no, est_imu, metrics, path):
    """(A) 真の軌道 vs 観測のみ / 観測+IMU の推定軌道を 3D で比較する図。"""
    fig = plt.figure(figsize=(11, 5.5))
    panels = [
        (Lbl("観測のみ", "obs only"), est_no, "gray",
         metrics["rmse_total_no_imu_mm"]),
        (Lbl("観測 + IMU 拘束", "obs + IMU"), est_imu, "tab:blue",
         metrics["rmse_total_with_imu_mm"]),
    ]
    for j, (title, est, col, rmse) in enumerate(panels):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.8,
                label=Lbl("真の軌道", "true"))
        ax.scatter(est[:, 0], est[:, 1], est[:, 2], c=col, s=22,
                   label=Lbl("推定", "estimate"))
        for t in range(len(traj)):     # 真値→推定の誤差線
            ax.plot([traj[t, 0], est[t, 0]], [traj[t, 1], est[t, 1]],
                    [traj[t, 2], est[t, 2]], color=col, lw=0.5, alpha=0.5)
        ax.set_title("%s\nRMSE %.0f mm" % (title, rmse), fontsize=10)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.legend(fontsize=8, loc="upper left")
        ax.view_init(elev=40, azim=-65)
    fig.suptitle(Lbl(
        "(A) 軌道推定: IMU 拘束で RMSE %.0f→%.0f mm (改善 %.0f%%)" % (
            metrics["rmse_total_no_imu_mm"], metrics["rmse_total_with_imu_mm"],
            metrics["imu_improvement_pct"]),
        "(A) Trajectory: IMU constraint %.0f->%.0f mm" % (
            metrics["rmse_total_no_imu_mm"], metrics["rmse_total_with_imu_mm"])))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_pointcloud(true_cloud, avg, metrics, path):
    """(B) 既知キューブの真の表面点 vs ステレオ推定点群を 3D で比較する図。"""
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(true_cloud[:, 0], true_cloud[:, 1], true_cloud[:, 2],
               c="red", s=8, alpha=0.5, label=Lbl("真の表面点", "true surface"))
    ax.scatter(avg[:, 0], avg[:, 1], avg[:, 2],
               c="tab:blue", s=10, alpha=0.7,
               label=Lbl("ステレオ推定点群", "stereo estimate"))
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.legend(fontsize=9, loc="upper left")
    ax.view_init(elev=22, azim=-60)
    fig.suptitle(Lbl(
        "(B) キューブ計測: 子機ステレオ looks=%d  点群RMS %.0f mm / "
        "寸法誤差(ロバスト) %+.0f mm" % (
            metrics["looks"], metrics["cloud_rms_mm"],
            metrics["robust_dim_error_mm"]),
        "(B) Cube: stereo looks=%d  cloud RMS %.0f mm" % (
            metrics["looks"], metrics["cloud_rms_mm"])))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _stereo_cloud(true_cloud, center, seed, looks,
                  standoff=STEREO_STANDOFF, baseline=STEREO_BASELINE,
                  sigma_cam=STEREO_SIGMA_CAM, up=STEREO_UP):
    """子機ステレオ(2カメラ)で各表面点を looks 回観測→三角測量し平均する (MATH_SPEC §6.2)。

    三角測量 (stereo_triangulate) には truth を渡さず、観測(方位)と既知カメラ位置のみ
    を入力する (MBD)。looks は撮影フレーム数 (多視点/時間平均)。
    """
    est = np.empty_like(true_cloud)
    for i, p in enumerate(true_cloud):
        c_L, c_R = stereo_camera_positions(p, center, standoff, baseline, up=up)
        acc = np.zeros(3)
        for m in range(looks):
            brg = simulate_stereo_observation(p, c_L, c_R, sigma_cam,
                                              seed=seed + i * 1000 + m)
            acc += stereo_triangulate(brg, c_L, c_R)
        est[i] = acc / looks
    return est


def part_a_trajectory(seed=0):
    print("=== (A) 軌道推定 (MATH_SPEC §5) ===")
    traj = double_lawnmower_trajectory()
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    z = apply_attitude_error_config(z, seed=seed)        # §14 波動揺 (config [attitude].as_error。既定 OFF)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 9999)

    est_no = estimate_trajectory(z, SIGMA, p_parent=P_PARENT)
    est_imu = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT)
    r_no = rmse_xyz(traj, est_no)
    r_imu = rmse_xyz(traj, est_imu)

    print(f"軌道点数 n           = {len(traj)} (ダブル芝刈り)")
    print(f"RMSE total  IMUなし  = {r_no['total']*1000:6.1f} mm")
    print(f"RMSE total  IMUあり  = {r_imu['total']*1000:6.1f} mm")
    improve = (1 - r_imu['total'] / r_no['total']) * 100
    print(f"IMU 拘束による改善    = {improve:5.1f} %")
    metrics = {
        "n_points": int(len(traj)),
        "rmse_total_no_imu_mm": r_no["total"] * 1000,
        "rmse_total_with_imu_mm": r_imu["total"] * 1000,
        "imu_improvement_pct": improve,
    }
    return traj, est_no, est_imu, metrics


def part_b_geometry(seed=0, n_per_edge=CUBE_N_PER_EDGE, looks=MAP_LOOKS):
    print("\n=== (B) 既知キューブのジオメトリ評価: 子機ステレオ2カメラ (MATH_SPEC §6.2) ===")
    L_true = CUBE_SIDE
    V_true = CUBE_SIDE**3
    true_cloud = true_cube_pointcloud(n_per_edge=n_per_edge)
    sig_deg = np.rad2deg(STEREO_SIGMA_CAM)
    print(f"真値 L_true / V_true = {L_true:.3f} m / {V_true*1e3:.2f} L  "
          f"(表面点 {len(true_cloud)} 点)")
    print(f"ステレオ: ベースライン {STEREO_BASELINE*100:.0f} cm / 観測距離(standoff) "
          f"{STEREO_STANDOFF:.1f} m / カメラ角度ノイズ {sig_deg:.2f} deg")

    # --- 単フレーム (looks=1): ステレオ奥行きノイズが効く厳しいケース ---
    one = _stereo_cloud(true_cloud, CUBE_CENTER, seed=seed, looks=1)
    print(f"\n[単フレーム looks=1]  点群RMS = "
          f"{pointcloud_rms_to_surface(one, true_cloud)*1000:.1f} mm")
    print(f"  AABB    L_hat = {cube_side_estimate(one):.3f} m  "
          f"寸法誤差 {dimension_error_mm(cube_side_estimate(one), L_true):+.0f} mm  "
          f"体積誤差率 {volume_error_rate_pct(aabb_volume(one), V_true):+.0f} %")

    # --- 多フレーム平均 (looks=N): 撮影フレームを平均化 ---
    avg = _stereo_cloud(true_cloud, CUBE_CENTER, seed=seed, looks=looks)
    rms = pointcloud_rms_to_surface(avg, true_cloud)
    Lr = robust_cube_side_estimate(avg)
    Vr = robust_volume(avg)
    print(f"\n[多フレーム平均 looks={looks}]  点群RMS = {rms*1000:.1f} mm")
    print(f"  AABB     L_hat = {cube_side_estimate(avg):.3f} m  "
          f"寸法誤差 {dimension_error_mm(cube_side_estimate(avg), L_true):+.0f} mm  "
          f"体積誤差率 {volume_error_rate_pct(aabb_volume(avg), V_true):+.0f} %")
    print(f"  ロバスト L_hat = {Lr:.3f} m  "
          f"寸法誤差 {dimension_error_mm(Lr, L_true):+.0f} mm  "
          f"体積誤差率 {volume_error_rate_pct(Vr, V_true):+.0f} %")
    print("注: ステレオ奥行き誤差 ~ Z^2*sigma/B。子機が接近(standoff小)・ベースライン大・"
          "多フレーム平均 + ロバスト寸法で実用域に入る。")
    metrics = {
        "L_true_m": L_true,
        "V_true_L": V_true * 1e3,
        "n_surface_points": int(len(true_cloud)),
        "looks": int(looks),
        "stereo_baseline_m": STEREO_BASELINE,
        "stereo_standoff_m": STEREO_STANDOFF,
        "stereo_sigma_cam_deg": sig_deg,
        "robust_L_hat_m": Lr,
        "robust_dim_error_mm": dimension_error_mm(Lr, L_true),
        "robust_vol_error_pct": volume_error_rate_pct(Vr, V_true),
        "cloud_rms_mm": rms * 1000,
    }
    return true_cloud, avg, metrics


def main(seed=SEED, export=True):
    traj, est_no, est_imu, traj_metrics = part_a_trajectory(seed=seed)
    true_cloud, avg, geom_metrics = part_b_geometry(seed=seed)
    if export:
        traj_png = os.path.join(FIGDIR, "trajectory.png")
        cloud_png = os.path.join(FIGDIR, "pointcloud.png")
        _plot_trajectory(traj, est_no, est_imu, traj_metrics, traj_png)
        _plot_pointcloud(true_cloud, avg, geom_metrics, cloud_png)
        write_json(
            "mapping/run_mapping",
            {"trajectory": traj_metrics, "geometry": geom_metrics},
            meta={"seed": int(seed), "script": "run_mapping.py"})
        write_report(
            "mapping", "Stage2 マッピング (軌道推定 + キューブ計測)",
            "Stage 2 を1本通す。(A) ダブル芝刈り軌道を観測のみ / 観測+IMU で推定し RMSE を比較\n"
            "(IMU拘束で精度向上)。(B) 既知キューブ表面を子機の2カメラ(ステレオ)で観測し三角測量で\n"
            "推定点群を作り、寸法 L_hat / 体積 V_hat と誤差を出す。位置推定(親機カメラ+音響)とは別系統。",
            condition_sections=["noise", "trajectory", "cube", "stereo", "mapping", "attitude"],
            not_reflected=[
                ("`[error_model]`/`[acoustic]`/`[sync]`",
                 "本シナリオは Stage2 の**能力デモ** (IMU拘束で軌道精度向上 / ステレオでキューブ計測)。"
                 "理想ノイズで効果を明快に示す。§8 の系統誤差・外れ値込みの測位評価は "
                 "`run_spec`/`run_deepwater` を参照。"),
                ("`[optical]`", "光減衰モデルは使わない (一定σ)。減衰込み測位は `run_deepwater`。"),
                ("`[depth]`/`[sbl]`/`[attitude]`", "深度・SBL・親機姿勢は使わない (別シナリオ)。"),
            ],
            outputs=[("trajectory.png", "(A) 真の軌道 vs 観測のみ/観測+IMU の推定軌道"),
                     ("pointcloud.png", "(B) 真の表面点 vs ステレオ推定点群"),
                     ("run_mapping.json", "軌道RMSEとキューブ計測の数値")],
            results={"軌道RMSE IMUなし→あり":
                     f"{traj_metrics['rmse_total_no_imu_mm']:.0f} → "
                     f"{traj_metrics['rmse_total_with_imu_mm']:.0f} mm",
                     "キューブ寸法誤差 (ロバスト)":
                     f"{geom_metrics['robust_dim_error_mm']:+.0f} mm",
                     "点群RMS": f"{geom_metrics['cloud_rms_mm']:.0f} mm"},
            meta={"seed": seed}, math_spec="§5, §6.2")
        print("\n結果を保存: results/mapping/")
    print("\n完了。Stage 2: 複数時刻推定 + ジオメトリ評価が一通り動作。")


if __name__ == "__main__":
    main()
