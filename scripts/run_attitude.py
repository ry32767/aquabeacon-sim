"""run_attitude.py — 親機姿勢と IMU 姿勢推定 (MATH_SPEC §14) を1本通すスクリプト。

親機は水上で波により不規則に動揺する。機体固定カメラの方位/仰角は機体フレームの値に
なるので、そのままワールド角度として位置推定すると姿勢ぶんの系統誤差が乗る。これを親機
搭載 IMU (ジャイロ+加速度+磁気) の SO(3) 相補フィルタで姿勢推定し、機体角度をワールドへ
補正する効果を示す。

3 条件で軌道 RMSE を比較する:
  (1) 動揺なし (baseline)        : 親機が静止 (R=I) の理想。
  (2) 動揺あり・補正なし (naive) : 機体角度をワールド角度と誤認 -> 姿勢誤差が位置に乗る。
  (3) 動揺あり・IMU補正 (corrected): 相補フィルタ姿勢で機体角度を補正 -> baseline へ回復。

フロー (MBD):
  (1) truth     : 真の軌道 + 波による親機姿勢の真値 R_true(t)
  (2) sensors   : 機体フレーム観測 (角度は R^T で回転) + IMU 生信号 (ジャイロ/加速度/磁気)
  (3) attitude  : IMU 信号のみから相補フィルタで姿勢 R_est を推定 (truth 非参照)
  (4) estimator : 補正後のワールド観測から軌道を推定 (truth 非参照)
  (5) evaluation: 真値 vs 推定で姿勢 RMS・位置 RMSE

乱数は seed 固定で再現可能。
"""
import os
import sys

import numpy as np

from _plotstyle import plt, USE_JP, Lbl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, P_PARENT, SEED, ATT_DT, ATT_GYRO_SIGMA,
                        ATT_GYRO_BIAS, ATT_ACC_SIGMA, ATT_MAG_SIGMA,
                        ATT_GRAVITY, ATT_FILTER_ALPHA, ATT_ROLL_AMP,
                        ATT_PITCH_AMP, ATT_YAW_AMP)
from src.truth import double_lawnmower_trajectory, wave_attitude_sequence
from src.sensors import (simulate_observation_sequence_attitude,
                         simulate_imu_signals)
from src.attitude import (euler_to_matrix, complementary_filter, euler_sequence,
                          correct_observation_sequence)
from src.estimator import estimate_trajectory
from src.evaluation import rmse_xyz
from src.results_io import write_json, write_report, scenario_dir

FIGDIR = scenario_dir("attitude")
AXES = ["roll", "pitch", "yaw"]


def _wrap(a):
    """角度差を (-pi, pi] に正規化。"""
    return np.arctan2(np.sin(a), np.cos(a))


def _matrices(euler):
    """Euler 列 (n,3) -> 回転行列列 (n,3,3)。"""
    return np.array([euler_to_matrix(*e) for e in euler])


def _plot_attitude(t, e_true, e_est, att_rms_deg, path):
    """roll/pitch/yaw の真値 vs IMU 相補フィルタ推定を時系列で比較する図。"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(t, np.rad2deg(e_true[:, i]), "-", color="red", lw=1.6,
                label=Lbl("真値", "true"))
        ax.plot(t, np.rad2deg(e_est[:, i]), "--", color="tab:blue", lw=1.3,
                label=Lbl("IMU推定", "IMU est"))
        ax.set_ylabel("%s [deg]" % AXES[i])
        ax.set_title(Lbl("%s  RMS誤差 %.3f deg", "%s  RMS err %.3f deg")
                     % (AXES[i], att_rms_deg[i]), fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    axs[-1].set_xlabel(Lbl("時刻 [s]", "time [s]"))
    fig.suptitle(Lbl(
        "(A) 親機姿勢: 波による動揺を IMU(ジャイロ+加速度+磁気) で推定  総RMS %.3f deg",
        "(A) Parent attitude: wave sway estimated by IMU  total RMS %.3f deg")
        % att_rms_deg[3])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_position(traj, est_naive, est_corr, metrics, path):
    """動揺・補正なし vs 動揺・IMU補正 の推定軌道を 3D で比較する図。"""
    fig = plt.figure(figsize=(11, 5.5))
    panels = [
        (Lbl("動揺あり・補正なし", "sway, no correction"), est_naive, "gray",
         metrics["rmse_naive_mm"]),
        (Lbl("動揺あり・IMU姿勢補正", "sway, IMU corrected"), est_corr, "tab:blue",
         metrics["rmse_corrected_mm"]),
    ]
    for j, (title, est, col, rmse) in enumerate(panels):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.8,
                label=Lbl("真の軌道", "true"))
        ax.scatter(est[:, 0], est[:, 1], est[:, 2], c=col, s=20,
                   label=Lbl("推定", "estimate"))
        for k in range(len(traj)):
            ax.plot([traj[k, 0], est[k, 0]], [traj[k, 1], est[k, 1]],
                    [traj[k, 2], est[k, 2]], color=col, lw=0.5, alpha=0.5)
        ax.set_title("%s\nRMSE %.0f mm" % (title, rmse), fontsize=10)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.legend(fontsize=8, loc="upper left")
        ax.view_init(elev=40, azim=-65)
    fig.suptitle(Lbl(
        "(B) 位置推定: 姿勢補正で RMSE %.0f->%.0f mm (baseline %.0f mm)" % (
            metrics["rmse_naive_mm"], metrics["rmse_corrected_mm"],
            metrics["rmse_baseline_mm"]),
        "(B) Position: attitude correction %.0f->%.0f mm" % (
            metrics["rmse_naive_mm"], metrics["rmse_corrected_mm"])))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(seed=SEED, export=True):
    print("=== 親機姿勢 + IMU 姿勢推定 (MATH_SPEC §14) ===")
    # --- (1) truth: 軌道 + 波による親機姿勢 ---
    traj = double_lawnmower_trajectory()
    n = len(traj)
    t = np.arange(n) * ATT_DT
    e_true = wave_attitude_sequence(n, dt=ATT_DT, seed=seed + 100)
    R_true = _matrices(e_true)
    R_I = np.tile(np.eye(3), (n, 1, 1))
    print("軌道点数 n           = %d  (dt=%.3f s, %.1f Hz)" % (n, ATT_DT, 1.0 / ATT_DT))
    print("動揺振幅 roll/pitch/yaw = %.1f / %.1f / %.1f deg" % (
        np.rad2deg(ATT_ROLL_AMP), np.rad2deg(ATT_PITCH_AMP), np.rad2deg(ATT_YAW_AMP)))

    # --- (2) sensors: 機体観測 (動揺あり/なし) + IMU 生信号 ---
    z_body = simulate_observation_sequence_attitude(traj, R_true, SIGMA, seed=seed,
                                                    p_parent=P_PARENT)
    z_base = simulate_observation_sequence_attitude(traj, R_I, SIGMA, seed=seed,
                                                    p_parent=P_PARENT)  # baseline (R=I)
    imu = simulate_imu_signals(R_true, dt=ATT_DT, seed=seed + 200,
                               gyro_sigma=ATT_GYRO_SIGMA, gyro_bias=ATT_GYRO_BIAS,
                               acc_sigma=ATT_ACC_SIGMA, mag_sigma=ATT_MAG_SIGMA,
                               gravity=ATT_GRAVITY)

    # --- (3) attitude: IMU 信号のみから相補フィルタで姿勢推定 (truth 非参照) ---
    R_est = complementary_filter(imu["gyro"], imu["acc"], imu["mag"], dt=ATT_DT,
                                 alpha=ATT_FILTER_ALPHA)
    e_est = euler_sequence(R_est)
    att_err = _wrap(e_est - e_true)
    att_rms_deg = [float(np.rad2deg(np.sqrt(np.mean(att_err[:, i] ** 2))))
                   for i in range(3)]
    att_rms_deg.append(float(np.rad2deg(np.sqrt(np.mean(att_err ** 2)))))  # total
    print("\n[姿勢推定] RMS誤差 roll/pitch/yaw = %.3f / %.3f / %.3f deg (総 %.3f)" % (
        att_rms_deg[0], att_rms_deg[1], att_rms_deg[2], att_rms_deg[3]))

    # --- (4) estimator: 3 条件で軌道推定 (truth 非参照) ---
    z_corr = correct_observation_sequence(z_body, R_est)
    est_base = estimate_trajectory(z_base, SIGMA, p_parent=P_PARENT)
    est_naive = estimate_trajectory(z_body, SIGMA, p_parent=P_PARENT)
    est_corr = estimate_trajectory(z_corr, SIGMA, p_parent=P_PARENT)

    # --- (5) evaluation ---
    r_base = rmse_xyz(traj, est_base)["total"] * 1000
    r_naive = rmse_xyz(traj, est_naive)["total"] * 1000
    r_corr = rmse_xyz(traj, est_corr)["total"] * 1000
    print("\n[位置 RMSE]")
    print("  (1) 動揺なし baseline       = %6.1f mm" % r_base)
    print("  (2) 動揺あり・補正なし naive = %6.1f mm" % r_naive)
    print("  (3) 動揺あり・IMU補正        = %6.1f mm" % r_corr)
    recov = (1 - (r_corr - r_base) / max(r_naive - r_base, 1e-9)) * 100
    print("  補正による誤差回復率         = %5.1f %% (naive の超過分を baseline へ戻した割合)" % recov)
    print("注: 機体カメラ角度を姿勢無視でワールド角度とすると yaw 誤差が方位角に直接効く。")
    print("    IMU 相補フィルタ(磁気=方位基準)で姿勢を推定し補正すると baseline 付近へ回復する。")

    metrics = {
        "n_points": int(n),
        "dt_s": ATT_DT,
        "roll_amp_deg": float(np.rad2deg(ATT_ROLL_AMP)),
        "pitch_amp_deg": float(np.rad2deg(ATT_PITCH_AMP)),
        "yaw_amp_deg": float(np.rad2deg(ATT_YAW_AMP)),
        "att_rms_roll_deg": att_rms_deg[0],
        "att_rms_pitch_deg": att_rms_deg[1],
        "att_rms_yaw_deg": att_rms_deg[2],
        "att_rms_total_deg": att_rms_deg[3],
        "rmse_baseline_mm": r_base,
        "rmse_naive_mm": r_naive,
        "rmse_corrected_mm": r_corr,
        "recovery_pct": float(recov),
    }

    if export:
        att_png = os.path.join(FIGDIR, "attitude.png")
        pos_png = os.path.join(FIGDIR, "position.png")
        _plot_attitude(t, e_true, e_est, att_rms_deg, att_png)
        _plot_position(traj, est_naive, est_corr, metrics, pos_png)
        write_json("attitude/run_attitude", metrics,
                   meta={"seed": int(seed), "script": "run_attitude.py"})
        write_report(
            "attitude", "親機姿勢と IMU 姿勢推定 (波による動揺の補正)",
            "親機は水上で波により不規則に動揺する。機体固定カメラの方位/仰角は機体フレームの\n"
            "値になるため、姿勢を無視すると位置推定に系統誤差 (特に yaw->方位角) が乗る。親機 IMU\n"
            "(ジャイロ+加速度+磁気) の SO(3) 相補フィルタで姿勢 R_est を推定し、機体角度をワールド\n"
            "へ補正する。動揺なし(baseline) / 補正なし(naive) / IMU補正 の3条件で軌道 RMSE を比較。",
            condition_sections=["attitude", "noise", "trajectory"],
            not_reflected=[
                ("`[error_model]`/`[acoustic]`/`[sync]`",
                 "機体観測は専用パス `simulate_observation_sequence_attitude` (§14) を通り、"
                 "現実誤差モデル (`simulate_observation_realistic`) とは別系統。バイアス・音速ズレ・"
                 "外れ値・遅延は本シナリオの機体角度観測には実装していない (姿勢推定の効果を切り分けるため)。"),
                ("`[optical]` (減衰σ)",
                 "親機光学リンクの減衰モデルは本シナリオの機体角度観測に適用しない (一定σ)。"),
                ("`[depth]`/`[sbl]`/`[stereo]`", "深度・SBL・ステレオは使わない (別シナリオ)。"),
            ],
            outputs=[("attitude.png", "(A) roll/pitch/yaw の真値 vs IMU 相補フィルタ推定"),
                     ("position.png", "(B) 補正なし vs IMU補正 の推定軌道 (vs 真の軌道)"),
                     ("run_attitude.json", "姿勢 RMS と位置 RMSE の数値")],
            results={"姿勢RMS (総)": "%.3f deg" % att_rms_deg[3],
                     "位置RMSE baseline->naive->補正":
                     "%.0f -> %.0f -> %.0f mm" % (r_base, r_naive, r_corr),
                     "誤差回復率": "%.1f %%" % recov},
            meta={"seed": seed}, math_spec="§14")
        print("\n結果を保存: results/attitude/")
    print("\n完了。§14: 波動揺の IMU 姿勢推定 + 機体角度のワールド補正が一通り動作。")
    return metrics


if __name__ == "__main__":
    main()
