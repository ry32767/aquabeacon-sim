"""
run_visualize.py — 発表用の図・アニメーションを生成する (可視化/評価レイヤ)。

MBD 上の位置づけ: ここは評価/プレゼン層なので、真値と推定値を突き合わせてよい。
ただし推定 (estimate_position) には truth を渡さず、観測のみを入力する原則は守る。

生成物はテストシナリオ (シーン) ごとに figures/ 以下のサブフォルダへ分けて出力する。
2系統 (測位=親機1カメラ / ジオメトリ=子機ステレオ) でグループ化:

  figures/positioning/        ← 親機カメラ + 音響での位置・軌道推定
    1_cloud3d/                3D 推定クラウド + 2σ誤差楕円体
    2_sensitivity/            感度解析 (距離/角度ノイズ/仰角)
    3_converge/               最小二乗 (単時刻) の収束
    4_trajectory/             芝刈り軌道の追従 (Stage2先取り)
    5_traj_imu/               複数時刻 軌道推定 IMU有無
    7_mapping_progress/       マッピング進行アニメ IMU有無
    9_traj_converge/          バンドル調整の収束
  figures/geometry/           ← 子機ステレオ2カメラでのジオメトリ計測
    6_cube_mapping/           キューブ計測 (ステレオ三角測量)
    8_multilook_converge/     多フレーム平均の収束
    10_stage2_sensitivity/    ステレオ感度 (standoff/baseline/frames)

各フォルダに PNG (+ アニメは GIF / MP4) が入る。

実行: python scripts/run_visualize.py
仕様: docs/VISUALIZATION.md
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # 画面なしでファイル保存するため
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (SIGMA, SIGMA_IMU, P_PARENT, CUBE_SIDE, CUBE_CENTER,
                        VIZ_CLOUD_N, VIZ_CUBE_N_PER_EDGE, VIZ_LOOKS,
                        VIZ_MAX_LOOKS, VIZ_ROTATE_FRAMES, VIZ_SENS_DISTS,
                        SENS_DEPTH_Z, SENS_ANGLE_DEGS, SENS_ELEV_DEGS,
                        SENS_NADIR_D, STEREO_BASELINE, STEREO_SIGMA_CAM,
                        STEREO_STANDOFF, SENS_STEREO_STANDOFFS, SENS_STEREO_BASELINES)
from src.truth import (true_child_position, demo_trajectory,
                       double_lawnmower_trajectory, true_cube_pointcloud)
from src.sensors import (forward_observation, simulate_observation,
                         inverse_observation, simulate_observation_sequence,
                         simulate_imu_displacements, stereo_camera_positions,
                         simulate_stereo_observation)
from src.estimator import (residual, h, weight_matrix, estimate_position,
                           estimate_trajectory)
from src.geometry import (aabb_dimensions, robust_cube_side_estimate,
                          robust_volume, aabb_volume, stereo_triangulate)
from src.evaluation import (monte_carlo_estimates, rmse_xyz, dimension_error_mm,
                            volume_error_rate_pct, pointcloud_rms_to_surface)

# ----------------------------------------------------------------------------
# 出力先・フォント・体裁
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# 日本語フォントを自動検出 (無ければ英語ラベルにフォールバックして豆腐□を防ぐ)
_JP_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "Hiragino Sans", "TakaoPGothic", "IPAexGothic"]
_available = {f.name for f in fm.fontManager.ttflist}
_JP = next((c for c in _JP_CANDIDATES if c in _available), None)
USE_JP = _JP is not None
if USE_JP:
    plt.rcParams["font.family"] = _JP
plt.rcParams["axes.unicode_minus"] = False    # マイナス記号の豆腐を防ぐ
plt.rcParams["figure.dpi"] = 110


def L(ja, en):
    """日本語フォントがあれば ja、無ければ en を返す (ラベル用)。"""
    return ja if USE_JP else en


def scene_dir(*parts):
    """シーン専用の出力サブフォルダ (figures/<group>/<scene>) を作って絶対パスを返す。

    テストシナリオごとに figures/ をフォルダ分けするためのヘルパー。
    例: scene_dir("positioning", "1_cloud3d") -> figures/positioning/1_cloud3d
    """
    d = os.path.join(FIGDIR, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def _save_anim(anim, basename, fps=15, outdir=None):
    """アニメを GIF (必須) と MP4 (ffmpeg があれば) で保存し、保存先一覧を返す。

    outdir を指定するとそのフォルダへ、未指定なら figures/ 直下へ保存する。
    """
    if outdir is None:
        outdir = FIGDIR
    saved = []
    gif = os.path.join(outdir, basename + ".gif")
    anim.save(gif, writer=PillowWriter(fps=fps))
    saved.append(gif)
    try:
        from matplotlib.animation import FFMpegWriter
        import shutil
        if shutil.which("ffmpeg"):
            mp4 = os.path.join(outdir, basename + ".mp4")
            anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=1800))
            saved.append(mp4)
    except Exception as e:        # MP4 は任意。失敗しても GIF があれば続行
        print(f"  (MP4 スキップ: {e})")
    return saved


def _set_3d_labels(ax, title):
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Y (North) [m]")
    ax.set_zlabel("Z (Up) [m]")
    ax.set_title(title)


# ----------------------------------------------------------------------------
# 収束デモ用: ガウス・ニュートン反復 (MATH_SPEC §4.2)。各反復の x を記録する。
#   x_{k+1} = x_k + (J^T W J)^{-1} J^T W r(x_k),  J = ∂h/∂x (数値ヤコビアン)
# estimate_position と同じ最適化を「途中経過を見せる」ために自前展開したもの。
# 入力は観測 z のみで、truth は参照しない。
# ----------------------------------------------------------------------------
def _numeric_jac_h(x, p_parent, eps=1e-6):
    J = np.zeros((3, 3))
    for j in range(3):
        dx = np.zeros(3)
        dx[j] = eps
        J[:, j] = (h(x + dx, p_parent) - h(x - dx, p_parent)) / (2 * eps)
    return J


def gauss_newton_path(z, sigma, x0, p_parent=None, n_iter=8):
    if p_parent is None:
        p_parent = np.zeros(3)
    W = weight_matrix(*sigma)
    x = np.asarray(x0, dtype=float).copy()
    xs = [x.copy()]
    for _ in range(n_iter):
        r = residual(x, z, p_parent)
        J = _numeric_jac_h(x, p_parent)
        delta = np.linalg.solve(J.T @ W @ J, J.T @ W @ r)
        x = x + delta
        xs.append(x.copy())
    return np.array(xs)


def _ellipsoid_surface(mean, cov, k=2.0, n=24):
    """共分散 cov の k シグマ楕円体の表面メッシュ (X,Y,Z) を返す。"""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    radii = k * np.sqrt(vals)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([sx, sy, sz], axis=-1) * radii        # スケール
    pts = pts @ vecs.T + mean                            # 回転 + 平行移動
    return pts[..., 0], pts[..., 1], pts[..., 2]


# ============================================================================
# Scene 1: 3D 推定クラウド + 2σ 誤差楕円体
# ============================================================================
def scene_cloud3d(n=VIZ_CLOUD_N, seed=0):
    print("scene1: 3D 推定クラウド ...")
    outdir = scene_dir("positioning", "1_cloud3d")
    truth = true_child_position()
    est = monte_carlo_estimates(truth, SIGMA, n=n, seed=seed, p_parent=P_PARENT)
    rmse = rmse_xyz(truth, est)
    mean = est.mean(axis=0)
    cov = np.cov(est.T)
    ex, ey, ez = _ellipsoid_surface(mean, cov, k=2.0)

    def draw(ax):
        ax.scatter(*P_PARENT, c="k", s=80, marker="^",
                   label=L("親機 (原点)", "Parent (origin)"))
        ax.scatter(est[:, 0], est[:, 1], est[:, 2], c="tab:blue", s=6,
                   alpha=0.25, label=L("推定 (N=%d)" % n, "Estimates (N=%d)" % n))
        ax.plot_surface(ex, ey, ez, color="tab:orange", alpha=0.18,
                        linewidth=0)
        ax.scatter(*truth, c="red", s=120, marker="*",
                   label=L("真値", "Truth"))
        # 親機から真値への視線
        ax.plot([P_PARENT[0], truth[0]], [P_PARENT[1], truth[1]],
                [P_PARENT[2], truth[2]], "k--", lw=0.8, alpha=0.5)
        _set_3d_labels(ax, L(
            "3D 推定クラウド  (RMSE total = %.1f mm, 楕円体=2σ)" % (rmse["total"] * 1000),
            "3D estimate cloud  (RMSE total = %.1f mm, ellipsoid=2sigma)" % (rmse["total"] * 1000)))
        ax.legend(loc="upper left", fontsize=8)

    # --- 静止画 ---
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax)
    ax.view_init(elev=18, azim=-60)
    png = os.path.join(outdir, "cloud3d.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # --- 回転 GIF/MP4 ---
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax)
    frames = VIZ_ROTATE_FRAMES

    def update(i):
        ax.view_init(elev=18, azim=-60 + i * (360 / frames))
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    saved = _save_anim(anim, "cloud3d_rotate", fps=18, outdir=outdir)
    plt.close(fig)
    print(f"  -> {png}")
    for s in saved:
        print(f"  -> {s}")
    print(f"  RMSE x/y/z/total [mm] = {rmse['x']*1000:.1f} / {rmse['y']*1000:.1f}"
          f" / {rmse['z']*1000:.1f} / {rmse['total']*1000:.1f}")


# ============================================================================
# Scene 2: 感度解析グラフ
# ============================================================================
def scene_sensitivity(n=VIZ_CLOUD_N):
    print("scene2: 感度解析グラフ ...")
    outdir = scene_dir("positioning", "2_sensitivity")
    from src.evaluation import monte_carlo_rmse

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) RMSE vs 距離
    depths = SENS_DEPTH_Z
    ds, rmses, theo = [], [], []
    for z in depths:
        truth = np.array([abs(z) * 0.6, abs(z) * 0.3, z])
        d = np.linalg.norm(truth)
        ds.append(d)
        rmses.append(monte_carlo_rmse(truth, SIGMA, n=n, seed=1)["total"] * 1000)
        theo.append(d * SIGMA[1] * 1000)
    axs[0].plot(ds, rmses, "o-", label=L("RMSE (推定)", "RMSE (estimate)"))
    axs[0].plot(ds, theo, "s--", color="gray",
                label=L("理論 d·σ_ang", "theory d·σ_ang"))
    axs[0].set_xlabel(L("親機-子機距離 d [m]", "range d [m]"))
    axs[0].set_ylabel("RMSE total [mm]")
    axs[0].set_title(L("(a) 距離 vs RMSE", "(a) range vs RMSE"))
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=8)

    # (b) RMSE vs 角度ノイズ
    truth = np.array([6.0, 8.0, -7.5])
    degs = SENS_ANGLE_DEGS
    rb = []
    for deg in degs:
        sig = (SIGMA[0], np.deg2rad(deg), np.deg2rad(deg))
        rb.append(monte_carlo_rmse(truth, sig, n=n, seed=2)["total"] * 1000)
    axs[1].plot(degs, rb, "o-", color="tab:red")
    axs[1].set_xlabel(L("角度ノイズ σ_ang [deg]", "angle noise σ_ang [deg]"))
    axs[1].set_ylabel("RMSE total [mm]")
    axs[1].set_title(L("(b) 角度精度 vs RMSE (d=12.5m)",
                       "(b) angle noise vs RMSE (d=12.5m)"))
    axs[1].grid(alpha=0.3)

    # (c) RMSE vs 仰角 (真下付近の破綻チェック)
    d = SENS_NADIR_D
    phis = SENS_ELEV_DEGS
    rc = []
    for pd in phis:
        phi = np.deg2rad(pd)
        truth = np.array([d * np.cos(phi), 0.0, d * np.sin(phi)])
        rc.append(monte_carlo_rmse(truth, SIGMA, n=n, seed=3)["total"] * 1000)
    axs[2].plot(phis, rc, "o-", color="tab:green")
    axs[2].set_xlabel(L("仰角 φ [deg] (-90=真下)", "elevation φ [deg] (-90=nadir)"))
    axs[2].set_ylabel("RMSE total [mm]")
    axs[2].set_title(L("(c) 仰角 vs RMSE (d=10m)", "(c) elevation vs RMSE (d=10m)"))
    axs[2].grid(alpha=0.3)

    fig.suptitle(L("感度解析: ノイズ・距離・仰角が位置精度に与える影響 (MATH_SPEC §7)",
                   "Sensitivity: noise / range / elevation vs position RMSE"))
    fig.tight_layout()
    png = os.path.join(outdir, "sensitivity.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png}")


# ============================================================================
# Scene 3: 最小二乗の収束アニメ
# ============================================================================
def scene_converge():
    print("scene3: 最小二乗の収束アニメ ...")
    outdir = scene_dir("positioning", "3_converge")
    truth = true_child_position()
    z = forward_observation(truth)               # ノイズフリー -> 真値に収束
    x0 = truth + np.array([5.0, -5.0, 5.0])      # 初期値を 5m ずらす
    path = gauss_newton_path(z, SIGMA, x0, p_parent=P_PARENT, n_iter=8)
    errs = np.linalg.norm(path - truth, axis=1) * 1000   # 各反復の誤差 [mm]

    def base(ax):
        ax.scatter(*P_PARENT, c="k", s=80, marker="^", label=L("親機", "Parent"))
        ax.scatter(*truth, c="red", s=150, marker="*", label=L("真値", "Truth"))
        ax.scatter(*x0, c="gray", s=60, marker="o", label=L("初期値", "Initial"))
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "-", color="tab:blue",
                lw=1.0, alpha=0.4)
        _set_3d_labels(ax, "")
        ax.legend(loc="upper left", fontsize=8)

    # 静止画 (収束パス全体)
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    base(ax)
    ax.scatter(path[:, 0], path[:, 1], path[:, 2], c=range(len(path)),
               cmap="viridis", s=40)
    ax.set_title(L("最小二乗の収束 (初期値→真値, %d反復で %.3f mm)" %
                   (len(path) - 1, errs[-1]),
                   "Least-squares convergence (%d iters, %.3f mm)" %
                   (len(path) - 1, errs[-1])))
    png = os.path.join(outdir, "converge.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # アニメ
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=-60)

    def update(i):
        ax.clear()
        base(ax)
        ax.view_init(elev=20, azim=-60)
        ax.scatter(path[:i + 1, 0], path[:i + 1, 1], path[:i + 1, 2],
                   c="tab:blue", s=40)
        ax.scatter(*path[i], c="tab:orange", s=120, marker="o",
                   edgecolors="k", label=L("現在の推定", "current"))
        ax.set_title(L("反復 %d / %d   誤差 = %.2f mm" % (i, len(path) - 1, errs[i]),
                       "iter %d / %d   error = %.2f mm" % (i, len(path) - 1, errs[i])))
        return ()

    anim = FuncAnimation(fig, update, frames=len(path), blit=False)
    saved = _save_anim(anim, "converge", fps=2, outdir=outdir)
    plt.close(fig)
    print(f"  -> {png}")
    for s in saved:
        print(f"  -> {s}")
    print(f"  収束誤差 [mm] = {np.round(errs, 3)}")


# ============================================================================
# Scene 4: 軌道追従アニメ (Stage 2 先取り)
# ============================================================================
def scene_trajectory(seed=100):
    print("scene4: 軌道追従アニメ (Stage2先取り) ...")
    outdir = scene_dir("positioning", "4_trajectory")
    traj = demo_trajectory(n_points=48)
    # 各時刻: ノイズ付き観測 -> 単時刻推定 (Stage1 を各点へ独立適用)
    est = np.empty_like(traj)
    for i, p in enumerate(traj):
        z = simulate_observation(p, SIGMA, seed=seed + i, p_parent=P_PARENT)
        est[i] = estimate_position(z, SIGMA, p_parent=P_PARENT)
    rmse = rmse_xyz(traj, est)

    lim = lambda a: (a.min() - 1, a.max() + 1)
    xl = lim(np.r_[traj[:, 0], est[:, 0], 0])
    yl = lim(np.r_[traj[:, 1], est[:, 1], 0])
    zl = lim(np.r_[traj[:, 2], est[:, 2], 0])

    def base(ax):
        ax.scatter(*P_PARENT, c="k", s=80, marker="^", label=L("親機", "Parent"))
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
        _set_3d_labels(ax, "")

    # 静止画 (全軌道 真値 vs 推定)
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    base(ax)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.5,
            label=L("真の軌道", "true path"))
    ax.scatter(est[:, 0], est[:, 1], est[:, 2], c="tab:blue", s=14,
               label=L("推定", "estimate"))
    for t, e in zip(traj, est):           # 誤差線
        ax.plot([t[0], e[0]], [t[1], e[1]], [t[2], e[2]], "gray", lw=0.4, alpha=0.5)
    ax.set_title(L("芝刈り軌道 追従 (RMSE total = %.1f mm)" % (rmse["total"] * 1000),
                   "Lawnmower tracking (RMSE total = %.1f mm)" % (rmse["total"] * 1000)))
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=35, azim=-65)
    png = os.path.join(outdir, "trajectory.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # アニメ (時間進行)
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        base(ax)
        ax.view_init(elev=35, azim=-65)
        ax.plot(traj[:i + 1, 0], traj[:i + 1, 1], traj[:i + 1, 2], "-",
                color="red", lw=1.5, label=L("真の軌道", "true path"))
        ax.scatter(est[:i + 1, 0], est[:i + 1, 1], est[:i + 1, 2],
                   c="tab:blue", s=14, label=L("推定", "estimate"))
        ax.plot([traj[i, 0], est[i, 0]], [traj[i, 1], est[i, 1]],
                [traj[i, 2], est[i, 2]], "tab:orange", lw=1.2)
        e_i = np.linalg.norm(est[i] - traj[i]) * 1000
        ax.set_title(L("t = %d / %d   瞬時誤差 = %.0f mm" % (i, len(traj) - 1, e_i),
                       "t = %d / %d   error = %.0f mm" % (i, len(traj) - 1, e_i)))
        ax.legend(loc="upper left", fontsize=8)
        return ()

    anim = FuncAnimation(fig, update, frames=len(traj), blit=False)
    saved = _save_anim(anim, "trajectory", fps=10, outdir=outdir)
    plt.close(fig)
    print(f"  -> {png}")
    for s in saved:
        print(f"  -> {s}")
    print(f"  追従 RMSE total [mm] = {rmse['total']*1000:.1f}")


# ============================================================================
# Scene 5: 複数時刻 軌道推定 (IMU 拘束あり/なし)  -- Stage 2 / MATH_SPEC §5
# ============================================================================
def scene_traj_imu(seed=0):
    print("scene5: 軌道推定 IMU有無の比較 (Stage2) ...")
    outdir = scene_dir("positioning", "5_traj_imu")
    traj = double_lawnmower_trajectory()
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 9999)
    est_no = estimate_trajectory(z, SIGMA, p_parent=P_PARENT)
    est_imu = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT)
    r_no = rmse_xyz(traj, est_no)["total"] * 1000
    r_imu = rmse_xyz(traj, est_imu)["total"] * 1000

    def draw(ax):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.8,
                label=L("真の軌道", "true path"))
        ax.scatter(est_no[:, 0], est_no[:, 1], est_no[:, 2], c="gray", s=14,
                   alpha=0.7, label=L("推定 IMUなし (%.0f mm)" % r_no,
                                      "no-IMU (%.0f mm)" % r_no))
        ax.scatter(est_imu[:, 0], est_imu[:, 1], est_imu[:, 2], c="tab:blue",
                   s=16, label=L("推定 IMUあり (%.0f mm)" % r_imu,
                                 "with-IMU (%.0f mm)" % r_imu))
        _set_3d_labels(ax, L(
            "複数時刻 軌道推定: IMU 拘束で RMSE %.0f→%.0f mm" % (r_no, r_imu),
            "Trajectory estimate: IMU %.0f->%.0f mm" % (r_no, r_imu)))
        ax.legend(loc="upper left", fontsize=8)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax)
    ax.view_init(elev=40, azim=-65)
    png = os.path.join(outdir, "traj_imu.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax)
    frames = VIZ_ROTATE_FRAMES

    def update(i):
        ax.view_init(elev=40, azim=-65 + i * (360 / frames))
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    saved = _save_anim(anim, "traj_imu_rotate", fps=18, outdir=outdir)
    plt.close(fig)
    print(f"  -> {png}")
    for s in saved:
        print(f"  -> {s}")
    print(f"  RMSE total [mm]: IMUなし {r_no:.0f} / IMUあり {r_imu:.0f}")


# ============================================================================
# Scene 6: キューブ マッピング (真の表面 vs 多視点平均の推定点群) -- §6
# ============================================================================
def scene_cube_mapping(seed=0, n_per_edge=VIZ_CUBE_N_PER_EDGE, looks=VIZ_LOOKS):
    print("scene6: キューブ マッピング (子機ステレオ, Stage2) ...")
    outdir = scene_dir("geometry", "6_cube_mapping")
    true_cloud = true_cube_pointcloud(n_per_edge=n_per_edge)
    est = _stereo_per_look(true_cloud, CUBE_CENTER, seed=seed,
                           max_looks=looks).mean(axis=1)

    L_true, V_true = CUBE_SIDE, CUBE_SIDE**3
    Lr = robust_cube_side_estimate(est)
    Vr = robust_volume(est)
    de = dimension_error_mm(Lr, L_true)
    ve = volume_error_rate_pct(Vr, V_true)
    rms = pointcloud_rms_to_surface(est, true_cloud) * 1000

    def draw(ax):
        ax.scatter(true_cloud[:, 0], true_cloud[:, 1], true_cloud[:, 2],
                   c="red", s=10, alpha=0.5, label=L("真の表面", "true surface"))
        ax.scatter(est[:, 0], est[:, 1], est[:, 2], c="tab:blue", s=10,
                   alpha=0.5, label=L("ステレオ推定 (×%dフレーム)" % looks,
                                      "stereo (x%d)" % looks))
        _set_3d_labels(ax, L(
            "キューブ計測: L誤差 %+.0f mm, 体積誤差 %+.0f%%, 点群RMS %.0f mm"
            % (de, ve, rms),
            "Cube: dimErr %+.0f mm, volErr %+.0f%%, RMS %.0f mm" % (de, ve, rms)))
        ax.legend(loc="upper left", fontsize=8)
        ax.set_box_aspect((1, 1, 1))

    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax)
    ax.view_init(elev=20, azim=-60)
    png = os.path.join(outdir, "cube_mapping.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax)
    frames = VIZ_ROTATE_FRAMES

    def update(i):
        ax.view_init(elev=20, azim=-60 + i * (360 / frames))
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    saved = _save_anim(anim, "cube_mapping_rotate", fps=18, outdir=outdir)
    plt.close(fig)
    print(f"  -> {png}")
    for s in saved:
        print(f"  -> {s}")
    print(f"  L誤差 {de:+.0f} mm / 体積誤差 {ve:+.0f}% / 点群RMS {rms:.0f} mm")


# ----------------------------------------------------------------------------
# Stage 2 動的シーン用の共通ヘルパー
# ----------------------------------------------------------------------------
def _stereo_per_look(true_cloud, center, seed, max_looks,
                     standoff=STEREO_STANDOFF, baseline=STEREO_BASELINE,
                     sigma_cam=STEREO_SIGMA_CAM):
    """各表面点を子機ステレオで max_looks フレーム観測→三角測量し (N, max_looks, 3) で返す。

    三角測量 (geometry) には truth を渡さず、観測(方位)と既知カメラ位置のみを入力 (MBD)。
    累積平均をとると looks=k の多フレーム平均になる (MATH_SPEC §6.2)。
    """
    n = len(true_cloud)
    out = np.empty((n, max_looks, 3))
    for i, p in enumerate(true_cloud):
        c_L, c_R = stereo_camera_positions(p, center, standoff, baseline)
        for m in range(max_looks):
            brg = simulate_stereo_observation(p, c_L, c_R, sigma_cam,
                                              seed=seed + i * 1000 + m)
            out[i, m] = stereo_triangulate(brg, c_L, c_R)
    return out


def _gauss_newton_trajectory_path(z_seq, sigma_obs, x0, imu=None,
                                  sigma_imu=None, p_parent=None, n_iter=12):
    """複数時刻バンドル調整 (MATH_SPEC §5) をガウス・ニュートンで解き、各反復の軌道を記録。

    estimate_trajectory と同じ残差を、収束過程を見せるために自前展開したもの。
    入力は観測 z_seq (と IMU) のみ。truth は参照しない。
    戻り値: (n_iter+1, n, 3)
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    z_seq = np.asarray(z_seq, float)
    n = len(z_seq)
    sqrtW_obs = 1.0 / np.asarray(sigma_obs, float)
    use_imu = imu is not None
    if use_imu:
        sqrtW_imu = 1.0 / np.broadcast_to(np.asarray(sigma_imu, float), (3,))

    def stacked(xflat):
        X = xflat.reshape(n, 3)
        parts = [sqrtW_obs * residual(X[k], z_seq[k], p_parent) for k in range(n)]
        if use_imu:
            for k in range(n - 1):
                parts.append(sqrtW_imu * ((X[k + 1] - X[k]) - imu[k]))
        return np.concatenate(parts)

    x = np.asarray(x0, float).ravel().copy()
    path = [x.reshape(n, 3).copy()]
    eps = 1e-6
    for _ in range(n_iter):
        r = stacked(x)
        J = np.zeros((len(r), len(x)))
        for j in range(len(x)):
            dx = np.zeros(len(x))
            dx[j] = eps
            J[:, j] = (stacked(x + dx) - stacked(x - dx)) / (2 * eps)
        delta = np.linalg.solve(J.T @ J + 1e-9 * np.eye(len(x)), J.T @ r)
        x = x - delta
        path.append(x.reshape(n, 3).copy())
    return np.array(path)


# ============================================================================
# Scene 7: マッピング進行アニメ (IMU 有無)  -- Stage 2 / MATH_SPEC §5
# ============================================================================
def scene_mapping_progress(seed=0):
    print("scene7: マッピング進行アニメ (IMU有無) ...")
    outdir = scene_dir("positioning", "7_mapping_progress")
    traj = double_lawnmower_trajectory(area=(6.0, 4.0), depth=-7.5,
                                       n_legs=3, pts_per_leg=6, origin=(3.0, 4.0))
    z = simulate_observation_sequence(traj, SIGMA, seed=seed, p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, SIGMA_IMU, seed=seed + 9999)
    est_no = estimate_trajectory(z, SIGMA, p_parent=P_PARENT)
    est_imu = estimate_trajectory(z, SIGMA, imu_deltas=imu, sigma_imu=SIGMA_IMU,
                                  p_parent=P_PARENT)
    r_no = rmse_xyz(traj, est_no)["total"] * 1000
    r_imu = rmse_xyz(traj, est_imu)["total"] * 1000
    lim = lambda a: (a.min() - 0.5, a.max() + 0.5)
    xl, yl, zl = lim(traj[:, 0]), lim(traj[:, 1]), lim(traj[:, 2])

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
        ax.view_init(elev=42, azim=-65)
        ax.plot(traj[:i + 1, 0], traj[:i + 1, 1], traj[:i + 1, 2], "-",
                color="red", lw=1.8, label=L("真の軌道", "true path"))
        ax.scatter(est_no[:i + 1, 0], est_no[:i + 1, 1], est_no[:i + 1, 2],
                   c="gray", s=14, alpha=0.7,
                   label=L("IMUなし (%.0f mm)" % r_no, "no-IMU (%.0f mm)" % r_no))
        ax.scatter(est_imu[:i + 1, 0], est_imu[:i + 1, 1], est_imu[:i + 1, 2],
                   c="tab:blue", s=16,
                   label=L("IMUあり (%.0f mm)" % r_imu, "with-IMU (%.0f mm)" % r_imu))
        _set_3d_labels(ax, L("マッピング進行 t=%d/%d" % (i, len(traj) - 1),
                             "mapping t=%d/%d" % (i, len(traj) - 1)))
        ax.legend(loc="upper left", fontsize=8)
        return ()

    update(len(traj) - 1)
    png = os.path.join(outdir, "mapping_progress.png")
    fig.savefig(png, bbox_inches="tight")
    anim = FuncAnimation(fig, update, frames=len(traj), blit=False)
    saved = _save_anim(anim, "mapping_progress", fps=10, outdir=outdir)
    plt.close(fig)
    print(f"  -> {png}")
    for s in saved:
        print(f"  -> {s}")
    print(f"  RMSE total [mm]: IMUなし {r_no:.0f} / IMUあり {r_imu:.0f}")


# ============================================================================
# Scene 8: 多視点平均の収束アニメ  -- Stage 2 / MATH_SPEC §6
# ============================================================================
def scene_multilook_converge(seed=0, n_per_edge=VIZ_CUBE_N_PER_EDGE,
                             max_looks=VIZ_MAX_LOOKS):
    print("scene8: 多フレーム平均の収束アニメ (子機ステレオ) ...")
    outdir = scene_dir("geometry", "8_multilook_converge")
    true_cloud = true_cube_pointcloud(n_per_edge=n_per_edge)
    per_look = _stereo_per_look(true_cloud, CUBE_CENTER, seed=seed,
                                max_looks=max_looks)
    csum = np.cumsum(per_look, axis=1)
    L_true, V_true = CUBE_SIDE, CUBE_SIDE**3
    lim = lambda c: (c.min() - 0.05, c.max() + 0.05)
    xl, yl, zl = lim(true_cloud[:, 0]), lim(true_cloud[:, 1]), lim(true_cloud[:, 2])

    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")

    def frame(k):
        looks = k + 1
        est = csum[:, k] / looks
        Lr = robust_cube_side_estimate(est)
        Vr = robust_volume(est)
        de = dimension_error_mm(Lr, L_true)
        ve = volume_error_rate_pct(Vr, V_true)
        rms = pointcloud_rms_to_surface(est, true_cloud) * 1000
        ax.clear()
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
        ax.view_init(elev=20, azim=-60)
        ax.scatter(true_cloud[:, 0], true_cloud[:, 1], true_cloud[:, 2],
                   c="red", s=8, alpha=0.4, label=L("真の表面", "true"))
        ax.scatter(est[:, 0], est[:, 1], est[:, 2], c="tab:blue", s=8,
                   alpha=0.6, label=L("推定", "estimate"))
        _set_3d_labels(ax, L(
            "ステレオ多フレーム平均 looks=%d:  L誤差 %+.0f mm, 体積誤差 %+.0f%%, RMS %.0f mm"
            % (looks, de, ve, rms),
            "stereo looks=%d: dimErr %+.0f mm, volErr %+.0f%%, RMS %.0f mm"
            % (looks, de, ve, rms)))
        ax.legend(loc="upper left", fontsize=8)
        return ()

    frame(max_looks - 1)
    png = os.path.join(outdir, "multilook_converge.png")
    fig.savefig(png, bbox_inches="tight")
    anim = FuncAnimation(fig, frame, frames=max_looks, blit=False)
    saved = _save_anim(anim, "multilook_converge", fps=6, outdir=outdir)
    plt.close(fig)
    est_final = csum[:, -1] / max_looks
    print(f"  -> {png}")
    for s in saved:
        print(f"  -> {s}")
    print(f"  looks={max_looks}: L誤差 "
          f"{dimension_error_mm(robust_cube_side_estimate(est_final), L_true):+.0f} mm")


# ============================================================================
# Scene 9: 軌道推定の収束アニメ (バンドル調整の反復) -- Stage 2 / MATH_SPEC §5
# ============================================================================
def scene_traj_converge(seed=0):
    print("scene9: 軌道推定の収束アニメ ...")
    outdir = scene_dir("positioning", "9_traj_converge")
    traj = double_lawnmower_trajectory(area=(6.0, 4.0), depth=-7.5,
                                       n_legs=2, pts_per_leg=5, origin=(3.0, 4.0))
    n = len(traj)
    z = np.array([forward_observation(p) for p in traj])   # ノイズフリー
    imu = np.diff(traj, axis=0)
    # 初期軌道: 逆変換解にランダムオフセットを足して「散らかった初期」を作る
    x0 = np.array([inverse_observation(*z[k]) for k in range(n)])
    rng = np.random.default_rng(seed)
    x0 = x0 + rng.normal(0, 1.5, x0.shape)
    path = _gauss_newton_trajectory_path(z, SIGMA, x0, imu=imu,
                                         sigma_imu=SIGMA_IMU, p_parent=P_PARENT)
    errs = np.array([rmse_xyz(traj, X)["total"] * 1000 for X in path])
    allp = np.vstack([traj, x0])
    lim = lambda c: (c.min() - 0.5, c.max() + 0.5)
    xl, yl, zl = lim(allp[:, 0]), lim(allp[:, 1]), lim(allp[:, 2])

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    def frame(i):
        ax.clear()
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
        ax.view_init(elev=40, azim=-65)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.8,
                label=L("真の軌道", "true path"))
        X = path[i]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c="tab:blue", s=22,
                   label=L("推定軌道", "estimate"))
        _set_3d_labels(ax, L("バンドル調整 反復 %d/%d   RMSE %.0f mm"
                             % (i, len(path) - 1, errs[i]),
                             "iter %d/%d  RMSE %.0f mm" % (i, len(path) - 1, errs[i])))
        ax.legend(loc="upper left", fontsize=8)
        return ()

    frame(len(path) - 1)
    png = os.path.join(outdir, "traj_converge.png")
    fig.savefig(png, bbox_inches="tight")
    anim = FuncAnimation(fig, frame, frames=len(path), blit=False)
    saved = _save_anim(anim, "traj_converge", fps=3, outdir=outdir)
    plt.close(fig)
    print(f"  -> {png}")
    for s in saved:
        print(f"  -> {s}")
    print(f"  RMSE/反復 [mm] = {np.round(errs, 1)}")


# ============================================================================
# Scene 10: Stage 2 指標の感度グラフ (距離・再観測回数)  -- §6
# ============================================================================
def scene_stage2_sensitivity(seed=0, n_per_edge=5):
    # n_per_edge は掃引で多数回推定するため速度優先の小さめ既定 (config とは独立)
    print("scene10: Stage2 ステレオ感度グラフ (距離・ベースライン・フレーム) ...")
    outdir = scene_dir("geometry", "10_stage2_sensitivity")
    L_true, V_true = CUBE_SIDE, CUBE_SIDE**3
    cloud = true_cube_pointcloud(n_per_edge=n_per_edge)
    looks_fixed = 15
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    def _metrics(est):
        return (abs(dimension_error_mm(robust_cube_side_estimate(est), L_true)),
                volume_error_rate_pct(robust_volume(est), V_true),
                pointcloud_rms_to_surface(est, cloud) * 1000)

    def _panel(ax, xs, de, ve, rms, xlabel, title):
        ax.plot(xs, de, "o-", label=L("寸法誤差 |L誤差|", "|dim err|"))
        ax.plot(xs, rms, "s--", color="gray", label=L("点群RMS", "cloud RMS"))
        ax.set_xlabel(xlabel); ax.set_ylabel("[mm]")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(xs, ve, "^:", color="tab:red", label=L("体積誤差率", "vol err %"))
        ax2.set_ylabel(L("体積誤差率 [%]", "vol err [%]"), color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    # (a) 観測距離 standoff を振る (baseline 既定, looks 固定)
    de_a, ve_a, rms_a = [], [], []
    for S in SENS_STEREO_STANDOFFS:
        est = _stereo_per_look(cloud, CUBE_CENTER, seed, looks_fixed,
                               standoff=S, baseline=STEREO_BASELINE).mean(axis=1)
        m = _metrics(est); de_a.append(m[0]); ve_a.append(m[1]); rms_a.append(m[2])
    _panel(axs[0], SENS_STEREO_STANDOFFS, de_a, ve_a, rms_a,
           L("観測距離 standoff [m]", "standoff [m]"),
           L("(a) 距離 vs 計測誤差 (B=%.0fcm)" % (STEREO_BASELINE * 100),
             "(a) standoff vs error"))

    # (b) ベースライン B を振る (standoff 既定, looks 固定)
    de_b, ve_b, rms_b = [], [], []
    for B in SENS_STEREO_BASELINES:
        est = _stereo_per_look(cloud, CUBE_CENTER, seed, looks_fixed,
                               standoff=STEREO_STANDOFF, baseline=B).mean(axis=1)
        m = _metrics(est); de_b.append(m[0]); ve_b.append(m[1]); rms_b.append(m[2])
    _panel(axs[1], [b * 100 for b in SENS_STEREO_BASELINES], de_b, ve_b, rms_b,
           L("ベースライン B [cm]", "baseline B [cm]"),
           L("(b) ベースライン vs 計測誤差 (Z=%.1fm)" % STEREO_STANDOFF,
             "(b) baseline vs error"))

    # (c) 撮影フレーム数 looks を振る (既定 standoff/baseline)
    per = _stereo_per_look(cloud, CUBE_CENTER, seed, VIZ_MAX_LOOKS)
    csum = np.cumsum(per, axis=1)
    looks_list = [v for v in [1, 2, 5, 10, 15, 20, 30] if v <= VIZ_MAX_LOOKS]
    de_c, ve_c, rms_c = [], [], []
    for k in looks_list:
        m = _metrics(csum[:, k - 1] / k)
        de_c.append(m[0]); ve_c.append(m[1]); rms_c.append(m[2])
    _panel(axs[2], looks_list, de_c, ve_c, rms_c,
           L("撮影フレーム数 looks", "num frames"),
           L("(c) フレーム数 vs 計測誤差", "(c) frames vs error"))

    fig.suptitle(L(
        "Stage 2 ステレオ感度: 接近(standoff小)・ベースライン大・多フレームで計測精度が上がる",
        "Stage 2 stereo sensitivity: closer / longer baseline / more frames -> better"))
    fig.tight_layout()
    png = os.path.join(outdir, "stage2_sensitivity.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png}")


def main():
    print(f"フォント: {_JP if USE_JP else '(日本語フォント無し -> 英語ラベル)'}")
    print(f"出力先  : {FIGDIR}\n")
    # Stage 1
    scene_cloud3d()
    scene_sensitivity()
    scene_converge()
    scene_trajectory()
    # Stage 2 (静止 + 回転)
    scene_traj_imu()
    scene_cube_mapping()
    # Stage 2 (動的: 進行・収束・感度)
    scene_mapping_progress()
    scene_multilook_converge()
    scene_traj_converge()
    scene_stage2_sensitivity()
    print("\n完了。figures/positioning/ と figures/geometry/ の各シーンフォルダに")
    print("PNG / GIF / MP4 が分かれて出力されました。発表資料に使ってください。")


if __name__ == "__main__":
    main()
