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

# =====================================================================
# 更新周期 (参考)
# =====================================================================
ACOUSTIC_RATE_HZ = float(_get("rates", "acoustic_hz", 5.0))
OPTICAL_RATE_HZ = float(_get("rates", "optical_hz", 30.0))

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
# 可視化 (run_visualize のシーン規模)
# =====================================================================
VIZ_CLOUD_N = int(_get("visualization", "cloud_n", 1500))
VIZ_CUBE_N_PER_EDGE = int(_get("visualization", "cube_n_per_edge", 7))
VIZ_LOOKS = int(_get("visualization", "looks", 20))
VIZ_MAX_LOOKS = int(_get("visualization", "max_looks", 30))
VIZ_ROTATE_FRAMES = int(_get("visualization", "rotate_frames", 72))
VIZ_SENS_DISTS = list(_get("visualization", "sens_dists", [3, 6, 9, 12.5, 16]))
