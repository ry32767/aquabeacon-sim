"""config.py — パラメータの一元管理 (MATH_SPEC §0, §7)。

実体の編集はリポジトリ直下の **config.toml** で行う (ユーザが触る場所)。
このモジュールは:
  1. 下のデフォルト値を定義し、
  2. config.toml が存在すればその値で上書きして、
  3. 各層が参照する定数 (SIGMA, P_PARENT, CUBE_SIDE ...) を公開する。

config.toml が無い / キーが欠けている場合はデフォルトが使われる (壊れない)。
他モジュールは値をハードコードせず、ここの定数を参照すること。

注意 (MBD): このファイルは真値定数 (TRUE_CHILD_POSITION, CUBE_CENTER) も持つ。
estimator / geometry はこのモジュールを import しないこと (truth を見ないため)。
"""
import os

import numpy as np

try:                       # tomllib は Python 3.11+ の標準ライブラリ (依存追加なし)
    import tomllib
except ModuleNotFoundError:
    tomllib = None

# --- config.toml を読み込む (あれば) -----------------------------------------
_TOML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.toml")
_CFG = {}
CONFIG_TOML_LOADED = False
if tomllib is not None and os.path.exists(_TOML_PATH):
    try:
        # utf-8-sig で読むことで、Windows のエディタが付ける BOM があっても除去する
        # (tomllib は BOM を受け付けないため。これが無いと黙ってデフォルトに戻る)
        with open(_TOML_PATH, "rb") as _f:
            _text = _f.read().decode("utf-8-sig")
        _CFG = tomllib.loads(_text)
        CONFIG_TOML_LOADED = True
    except Exception as _e:    # パース失敗時はデフォルトにフォールバック (警告のみ)
        print(f"[config] config.toml の読み込みに失敗しました: {_e} "
              f"-> デフォルト値を使用します")


def _get(section, key, default):
    """config.toml の [section] key を取得。無ければ default。"""
    return _CFG.get(section, {}).get(key, default)


# 生の config 辞書とパスを公開 (results_io がシナリオ説明 .md に条件を埋め込むのに使う)
CONFIG_RAW = _CFG
CONFIG_TOML_PATH = _TOML_PATH


# =====================================================================
# 座標系・幾何 (MATH_SPEC §0)
# =====================================================================
# ワールド座標系 W: 親機(水上)を原点とする ENU (East-North-Up)。X=東 Y=北 Z=上。
P_PARENT = np.asarray(_get("coords", "p_parent", [0.0, 0.0, 0.0]), dtype=float)
# 観測の向き (§0.3)。True: 親機→子機 (v=p_child-p_parent) / False: 子機→親機。
OBSERVE_FROM_PARENT = bool(_get("coords", "observe_from_parent", True))

# =====================================================================
# 観測ノイズ (MATH_SPEC §7。AquaBeacon 目標精度を初期値)
# =====================================================================
SIGMA_DIST = float(_get("noise", "sigma_dist", 0.03))            # 距離 [m]
SIGMA_AZ = np.deg2rad(float(_get("noise", "sigma_az_deg", 0.3)))  # 方位角 [rad]
SIGMA_EL = np.deg2rad(float(_get("noise", "sigma_el_deg", 0.3)))  # 仰角 [rad]
SIGMA = (SIGMA_DIST, SIGMA_AZ, SIGMA_EL)

# IMU 拘束 (Stage 2 / §5)。pre-integration の時刻間変位ノイズ [m/軸]。
SIGMA_IMU = float(_get("noise", "sigma_imu", 0.02))
SIGMA_IMU_VEC = (SIGMA_IMU, SIGMA_IMU, SIGMA_IMU)

# 光学↔フォールバック 自動切替 (MATH_SPEC §12)。
SWITCH_DROPOUT_THRESHOLD = float(_get("switch", "dropout_threshold", 0.2))
SWITCH_SNR_MARGIN = float(_get("switch", "snr_margin", 1.0))
SWITCH_HYSTERESIS = float(_get("switch", "hysteresis", 0.05))

# 深度センサ (圧力) による鉛直拘束 (MATH_SPEC §10)。
DEPTH_ENABLE = bool(_get("depth", "enable", False))
SIGMA_DEPTH = float(_get("depth", "sigma_m", 0.05))     # 深度ノイズ [m]
DEPTH_BIAS = float(_get("depth", "bias_m", 0.0))        # 深度バイアス [m]

# SBL: 親機4トランスデューサ音響測位 (MATH_SPEC §13)。
SBL_BASELINE = float(_get("sbl", "baseline", 4.0))      # アレイ一辺 [m]
SBL_SIGMA_RANGE = float(_get("sbl", "sigma_range", SIGMA_DIST))   # 各測距ノイズ [m]
# 親機 (原点, 水面) を中心に一辺 SBL_BASELINE の正方形 4 隅 (z=0) に配置。
_sbl_b = SBL_BASELINE / 2.0
SBL_ANCHORS = np.array([[_sbl_b, _sbl_b, 0.0], [_sbl_b, -_sbl_b, 0.0],
                        [-_sbl_b, _sbl_b, 0.0], [-_sbl_b, -_sbl_b, 0.0]], dtype=float)

# =====================================================================
# 更新周期 (参考)
# =====================================================================
ACOUSTIC_RATE_HZ = float(_get("rates", "acoustic_hz", 5.0))
OPTICAL_RATE_HZ = float(_get("rates", "optical_hz", 30.0))

# =====================================================================
# 現実的センサ誤差モデル (MATH_SPEC §8)
# 既定はすべて「理想」で、零平均ガウスの従来挙動と一致する (enable=False)。
# ERROR_MODEL は sensors.simulate_observation_realistic にそのまま渡せる辞書。
# =====================================================================
ERROR_MODEL_ENABLE = bool(_get("error_model", "enable", False))
ERROR_MODEL = {
    "bias": (
        float(_get("error_model", "bias_dist", 0.0)),
        np.deg2rad(float(_get("error_model", "bias_az_deg", 0.0))),
        np.deg2rad(float(_get("error_model", "bias_el_deg", 0.0))),
    ),
    "range_growth_per_m": float(_get("error_model", "range_growth_per_m", 0.0)),
    "dist_growth_per_m": float(_get("error_model", "dist_growth_per_m", 0.0)),
    "outlier_rate": float(_get("error_model", "outlier_rate", 0.0)),
    "outlier_scale": float(_get("error_model", "outlier_scale", 20.0)),
    "sound_speed_true": float(_get("acoustic", "sound_speed_true", 1500.0)),
    "sound_speed_assumed": float(_get("acoustic", "sound_speed_assumed", 1500.0)),
    "acoustic_latency_s": float(_get("sync", "acoustic_latency_s", 0.0)),
}

# =====================================================================
# ロバスト推定 (MATH_SPEC §4.4)。既定 'linear' は従来の純 L2 と一致。
# =====================================================================
EST_LOSS = str(_get("estimator", "loss", "linear"))
EST_F_SCALE = float(_get("estimator", "f_scale", 1.345))

# =====================================================================
# 親機の光学リンク: 水中の減衰・拡散モデル (MATH_SPEC §9)
# OPTICAL_MODEL は sensors の optical_* / simulate_observation_realistic に渡せる辞書
# (角度は rad、距離は m、c は 1/m)。既定は enable=False で従来挙動と一致。
# =====================================================================
OPTICAL_ENABLE = bool(_get("optical", "enable", False))
OPTICAL_MODEL = {
    "attenuation_c": float(_get("optical", "attenuation_c", 0.30)),
    "range_ref": float(_get("optical", "range_ref", 10.0)),
    "sigma_ref": np.deg2rad(float(_get("optical", "sigma_ref_deg", 0.3))),
    "sigma_floor": np.deg2rad(float(_get("optical", "sigma_floor_deg", 0.08))),
    "snr_ref": float(_get("optical", "snr_ref", 40.0)),
    "snr_exponent": float(_get("optical", "snr_exponent", 1.0)),
    "snr_min": float(_get("optical", "snr_min", 6.0)),
    "dropout_max": float(_get("optical", "dropout_max", 0.5)),
    "dropout_jump": np.deg2rad(float(_get("optical", "dropout_jump_deg", 30.0))),
}

# =====================================================================
# 親機姿勢と IMU 姿勢推定 (MATH_SPEC §14)
# 波で動揺する親機の姿勢を IMU (ジャイロ+加速度+磁気) で推定し、機体カメラ角度を補正する。
# 角度は config.toml では [deg]/[deg/s] で書き、ここで rad に変換する。
# =====================================================================
ATT_ENABLE = bool(_get("attitude", "enable", False))
ATT_DT = float(_get("attitude", "dt", 0.02))                  # サンプル間隔 [s] (50 Hz)
ATT_ROLL_AMP = np.deg2rad(float(_get("attitude", "roll_amp_deg", 5.0)))    # 動揺振幅 [rad]
ATT_PITCH_AMP = np.deg2rad(float(_get("attitude", "pitch_amp_deg", 4.0)))
ATT_YAW_AMP = np.deg2rad(float(_get("attitude", "yaw_amp_deg", 3.0)))
ATT_ROLL_PERIOD = float(_get("attitude", "roll_period_s", 4.0))           # 主要周期 [s]
ATT_PITCH_PERIOD = float(_get("attitude", "pitch_period_s", 5.0))
ATT_YAW_PERIOD = float(_get("attitude", "yaw_period_s", 8.0))
ATT_YAW_MEAN = np.deg2rad(float(_get("attitude", "yaw_mean_deg", 0.0)))    # 方位オフセット [rad]
ATT_GYRO_SIGMA = np.deg2rad(float(_get("attitude", "gyro_sigma_dps", 0.1)))   # [rad/s]
ATT_GYRO_BIAS = np.deg2rad(float(_get("attitude", "gyro_bias_dps", 0.05)))    # [rad/s]
ATT_ACC_SIGMA = float(_get("attitude", "acc_sigma", 0.05))    # 加速度ノイズ [m/s^2]
ATT_MAG_SIGMA = float(_get("attitude", "mag_sigma", 0.02))    # 磁気ノイズ [-]
ATT_GRAVITY = float(_get("attitude", "gravity", 9.80665))     # 重力 [m/s^2]
ATT_FILTER_ALPHA = float(_get("attitude", "filter_alpha", 0.98))   # 相補係数 (1=ジャイロのみ)

# 深い水深のテストシナリオ (run_deepwater.py)
DEEP_DEPTHS = list(_get("deepwater", "depths", [5, 10, 15, 20]))
DEEP_HORIZ_OFFSET = np.asarray(_get("deepwater", "horiz_offset", [3.0, 2.0]), dtype=float)
DEEP_CLARITIES = list(_get("deepwater", "clarities", [0.05, 0.3, 1.0]))
DEEP_TRAJ_DEPTH = float(_get("deepwater", "traj_depth", 13.0))
DEEP_TRAJ_CLARITY = float(_get("deepwater", "traj_clarity", 0.5))
DEEP_MC_N = int(_get("deepwater", "mc_n", 600))

# =====================================================================
# 再現性
# =====================================================================
SEED = int(_get("montecarlo", "seed", 0))
MC_N = int(_get("montecarlo", "n", 2000))     # モンテカルロ試行数

# =====================================================================
# Stage 1 の真値 (truth.py が参照。推定側は参照しないこと)
# =====================================================================
TRUE_CHILD_POSITION = np.asarray(
    _get("truth", "child_position", [6.0, 8.0, -7.5]), dtype=float)

# =====================================================================
# Stage 2 の真値: 既知物体 (キューブ)
# =====================================================================
CUBE_SIDE = float(_get("cube", "side", 0.5))
CUBE_CENTER = np.asarray(_get("cube", "center", [6.0, 8.0, -7.5]), dtype=float)
CUBE_N_PER_EDGE = int(_get("cube", "n_per_edge", 7))

# =====================================================================
# 子機の2カメラ (ステレオ) ジオメトリ (MATH_SPEC §6.2)
# =====================================================================
STEREO_BASELINE = float(_get("stereo", "baseline", 0.10))        # [m]
STEREO_SIGMA_CAM = np.deg2rad(float(_get("stereo", "sigma_cam_deg", 0.1)))  # [rad]
STEREO_STANDOFF = float(_get("stereo", "standoff", 2.0))         # [m]
STEREO_UP = np.asarray(_get("stereo", "up_axis", [0.0, 0.0, 1.0]), dtype=float)

# =====================================================================
# Stage 2 ダブル芝刈り軌道 (run_mapping)
# =====================================================================
TRAJ_AREA = tuple(_get("trajectory", "area", [8.0, 6.0]))
TRAJ_DEPTH = float(_get("trajectory", "depth", -7.5))
TRAJ_N_LEGS = int(_get("trajectory", "n_legs", 4))
TRAJ_PTS_PER_LEG = int(_get("trajectory", "pts_per_leg", 12))
TRAJ_ORIGIN = tuple(_get("trajectory", "origin", [2.0, 5.0]))

# =====================================================================
# Stage 1 可視化用の簡易芝刈り (scene4)
# =====================================================================
DEMO_N_POINTS = int(_get("demo_trajectory", "n_points", 60))
DEMO_AREA = tuple(_get("demo_trajectory", "area", [8.0, 6.0]))
DEMO_DEPTH = float(_get("demo_trajectory", "depth", -7.5))
DEMO_N_LEGS = int(_get("demo_trajectory", "n_legs", 4))
DEMO_ORIGIN = tuple(_get("demo_trajectory", "origin", [2.0, -3.0]))
DEMO_DEPTH_RIPPLE = float(_get("demo_trajectory", "depth_ripple", 0.6))

# =====================================================================
# マッピング (run_mapping のキューブ計測)
# =====================================================================
MAP_LOOKS = int(_get("mapping", "looks", 30))

# =====================================================================
# 感度解析 (run_sensitivity)
# =====================================================================
SENS_DEPTH_Z = list(_get("sensitivity", "depth_z", [-2, -5, -7.5, -10, -15]))
SENS_ANGLE_DEGS = list(_get("sensitivity", "angle_degs", [0.1, 0.3, 0.5, 1.0]))
SENS_ELEV_DEGS = list(_get("sensitivity", "elev_degs", [-89, -80, -60, -45, -20]))
SENS_NADIR_D = float(_get("sensitivity", "nadir_d", 10.0))
SENS_STEREO_STANDOFFS = list(_get("sensitivity", "stereo_standoffs", [1.0, 2.0, 3.0, 5.0, 8.0]))
SENS_STEREO_BASELINES = list(_get("sensitivity", "stereo_baselines", [0.05, 0.1, 0.2, 0.4]))

# =====================================================================
# 設計スペックシート (run_spec.py): 目標精度と探索グリッド
# =====================================================================
SPEC_POS_RMSE_TARGET_MM = float(_get("spec", "pos_rmse_target_mm", 100.0))
SPEC_OPDEPTH_TARGET_MM = float(_get("spec", "op_depth_target_mm", 300.0))
SPEC_POS_BEARING = np.asarray(_get("spec", "pos_bearing", [0.6, 0.3, -0.75]), dtype=float)
SPEC_POS_RANGE_GRID = list(_get("spec", "pos_range_grid", [5, 8, 10, 12.5, 15, 18, 22]))
SPEC_POS_NOMINAL_RANGE = float(_get("spec", "pos_nominal_range", 12.5))
SPEC_POS_SIGMA_ANG_GRID = list(_get("spec", "pos_sigma_ang_grid", [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]))
SPEC_GEOM_DIM_TARGET_MM = float(_get("spec", "geom_dim_target_mm", 30.0))
SPEC_GEOM_STANDOFF_GRID = list(_get("spec", "geom_standoff_grid", [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]))
SPEC_GEOM_BASELINE_GRID = list(_get("spec", "geom_baseline_grid", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]))
SPEC_GEOM_SIGMA_CAM_GRID = list(_get("spec", "geom_sigma_cam_grid", [0.03, 0.05, 0.1, 0.2, 0.3]))
SPEC_GEOM_LOOKS_GRID = list(_get("spec", "geom_looks_grid", [1, 3, 5, 10, 15, 20, 30]))
SPEC_GEOM_N_PER_EDGE = int(_get("spec", "geom_n_per_edge", 5))
SPEC_MC_N = int(_get("spec", "mc_n", 800))

# =====================================================================
# 可視化 (run_visualize のシーン規模)
# =====================================================================
VIZ_CLOUD_N = int(_get("visualization", "cloud_n", 1500))
VIZ_CUBE_N_PER_EDGE = int(_get("visualization", "cube_n_per_edge", 7))
VIZ_LOOKS = int(_get("visualization", "looks", 20))
VIZ_MAX_LOOKS = int(_get("visualization", "max_looks", 30))
VIZ_ROTATE_FRAMES = int(_get("visualization", "rotate_frames", 72))
VIZ_SENS_DISTS = list(_get("visualization", "sens_dists", [3, 6, 9, 12.5, 16]))
