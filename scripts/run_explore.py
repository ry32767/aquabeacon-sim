"""run_explore.py — 対話的パラメータ探索ツール (測位手法の比較)。

config.toml を編集せずに、**スライダ/チェックで条件を決め、[計算] ボタンで 3 つの測位手法の
「RMSE vs 水深」曲線を比較**するための探索 UI。同じ near-nadir サーベイ軌道 (config [survey]) を
共有し、水深グリッドを掃引して各手法の total RMSE を折れ線で重ねる。逐次計算は重いので、
スライダ/チェックは state を更新するだけで再計算せず、[計算] ボタンを押したときにだけ掃引する。

全誤差モデルをコード内 (UI/CLI) で変更可能:
  config.toml の **誤差まわり全パラメータ** (§7 ノイズ / §8 現実誤差 / §8.4 音速 / §8.5 同期 /
  §9 光減衰 / §10 深度 / §13 SBL / §4.4 ロバスト損失) を、config.toml を書き換えずにこのツールの
  中で上書きして結果を見られる。パラメータが多いので**種類ごとのページ**に整理し (左の「パラメータ群」
  ラジオで切替)、各ページ最大 5 本のスライダだけを表示して UI が破綻しないようにする。

実行のたびに実効設定を results/explore/config.toml に出力:
  UI/CLI で上書きした「最終的に使ったパラメータ」を **config.toml 形式**で results/explore/config.toml
  に書き出す。元の config.toml は変更しない。出力をリポジトリ直下にコピーすればその設定で再現できる。

比較する手法 (run_sbl と同じ系):
  - 光学      : 親機カメラ(角度) + 音響1点距離 + IMU (+任意で深度)        (§5/§9)
  - SBL       : 親機4トランスデューサへの距離(多辺測量) + IMU + 深度        (§13)
  - 光学なし  : 音響1点距離 + IMU + 深度 (方位を使わない可観測性)          (§11)
    ※ SBL と光学なしは構造上 深度が必須 (深度チェックは光学にのみ作用)。

高速化: (手法 x 水深 x seed) の全評価を 1 つのタスク列にまとめ、**CPU マルチプロセス並列**
(ProcessPoolExecutor) で実行する。各評価は独立した最小二乗推定で CPU バウンドなのでそのまま
スケールする。並列度は --workers で制御 (既定=CPUコア数、1 で逐次)。結果・再現性は逐次と同一。

GUI バックエンドが無い環境では `--once` 相当の一発計算にフォールバックし、結果(表)を表示して
results/explore/ に「RMSE vs 水深」曲線・config.toml・README を保存する。CLI でも全項目を上書き可:
    python scripts/run_explore.py --once --clarity 0.6 --sbl-baseline 4 --depths 5,10,15,20
    python scripts/run_explore.py --once --set outlier_rate=0.1 --set sound_speed_true=1520 --loss cauchy

MBD: 観測生成 (sensors) と推定 (estimator, 入力は観測のみ) と評価 (truth vs 推定) を分離。
推定には truth を渡さない。乱数は seed 固定で再現可能。
"""
import argparse
import copy
import multiprocessing as mp
import os
import sys
from datetime import datetime, timezone

import numpy as np

# --- インタラクティブ表示のため、pyplot より前にバックエンドを選ぶ -----------------
# (_plotstyle は Agg 固定なので import しない。GUI が無ければ Agg にフォールバック)
#  並列計算の子プロセス (ProcessPoolExecutor) ではこのモジュールが再 import される。
#  子では GUI バックエンドは不要 (図は親だけが描く) なので、Agg を強制して Tk/Qt の
#  読込コストと「DISPLAY なし」警告を避ける。プロセス名で親/子を判定する
#  (spawn のブートストラップは main モジュール再 import の前にプロセス名を設定する)。
import matplotlib
_IS_WORKER = mp.current_process().name != "MainProcess"
_INTERACTIVE = True
if _IS_WORKER:
    matplotlib.use("Agg")
    _INTERACTIVE = False
else:
    for _bk in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            matplotlib.use(_bk)
            break
        except Exception:
            continue
    else:
        matplotlib.use("Agg")
        _INTERACTIVE = False
import matplotlib.pyplot as plt                                          # noqa: E402
import matplotlib.font_manager as fm                                     # noqa: E402
from matplotlib.widgets import Slider, CheckButtons, Button, RadioButtons  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (P_PARENT, SIGMA_DIST, SIGMA_AZ, SIGMA_EL, SIGMA_IMU,
                        SIGMA_DEPTH, DEPTH_BIAS, DEPTH_ENABLE, SEED,
                        OPTICAL_MODEL, OPTICAL_ENABLE, ERROR_MODEL, ERROR_MODEL_ENABLE,
                        EST_LOSS, EST_F_SCALE, SBL_BASELINE, SBL_SIGMA_RANGE,
                        SURVEY_AREA, SURVEY_ORIGIN, CONFIG_RAW, CONFIG_TOML_LOADED,
                        ATT_AS_ERROR, ATT_IMU_CORRECT, ATT_DT, ATT_FILTER_ALPHA,
                        ATT_YAW_MEAN, ATT_IMU_KW, ATT_ROLL_AMP, ATT_PITCH_AMP,
                        ATT_YAW_AMP, ATT_ROLL_PERIOD, ATT_PITCH_PERIOD, ATT_YAW_PERIOD)
from src.rng import substream_seed
from src.truth import double_lawnmower_trajectory
from src.sensors import (simulate_observation_sequence,
                         simulate_observation_sequence_realistic,
                         simulate_imu_displacements, simulate_depth_sequence,
                         simulate_sbl_range_sequence, optical_angular_sigma,
                         apply_attitude_error, sbl_attitude_anchors)
from src.estimator import (estimate_trajectory, estimate_trajectory_sbl,
                           estimate_trajectory_acoustic_inertial)
from src.evaluation import rmse_xyz
from src.results_io import scenario_dir, write_report

# --- 日本語フォント検出 (_plotstyle と同じ候補。文字化け回避) -----------------------
#  並列ワーカーは図を描かないので、起動を速くするためフォント走査をスキップする
#  (ラベルは RMSE 値に影響しない)。親プロセスだけが検出して描画に使う。
if _IS_WORKER:
    _JP = None
else:
    _JP = next((c for c in ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                            "Hiragino Sans", "TakaoPGothic", "IPAexGothic"]
                if c in {f.name for f in fm.fontManager.ttflist}), None)
    if _JP:
        plt.rcParams["font.family"] = _JP
    plt.rcParams["axes.unicode_minus"] = False


def L(ja, en):
    return ja if _JP else en


# 応答性を優先した規模 (小さな near-nadir 箱 + 少数シード)。--n-seeds で変更可。
N_SEEDS = 2
N_LEGS, PTS_PER_LEG = 2, 4          # 軌道点数 = 8
NO_OPT_AZ = 6                       # 光学なしの方位スタート数 (速度優先)
DEPTH_GRID = [1, 5, 10, 15, 20, 25]  # 水深掃引 [m] (横軸, 1〜25m)。--depths で変更可

# CPU 並列計算のワーカー数 (main() で --workers から設定。None=自動=os.cpu_count())。
WORKERS = None

METHOD_KEYS = ["optical", "sbl", "no_optical"]
METHOD_LABEL = {"optical": L("光学", "optical"), "sbl": L("SBL", "SBL"),
                "no_optical": L("光学なし", "no-optical")}
METHOD_COLOR = {"optical": "tab:green", "sbl": "tab:blue", "no_optical": "tab:orange"}

LOSS_CHOICES = ["linear", "huber", "cauchy", "soft_l1", "arctan"]

# =====================================================================
# 調整可能パラメータの一元定義 (UI/CLI/config.toml 出力の単一の出典)
#   - key      : state 辞書のキー (CLI --set のキーでもある)
#   - ja/en    : スライダ表示名 (単位つき)
#   - lo/hi    : スライダ範囲 (初期値が外なら自動拡張)
#   - group    : パラメータ群 (UI のページ。GROUP_ORDER の順に左ラジオで切替)
#   - section  : config.toml のセクション名 (出力先)
#   - toml     : config.toml のキー名 (出力先)
#   - angle    : True なら state は [deg] で持ち、モデルへ渡す際に [rad] へ変換、
#                config.toml には *_deg キーで [deg] のまま書く
# 角度系の既定値は config が [rad] で保持しているので [deg] へ戻して初期値とする。
# =====================================================================
GROUP_ORDER = ["noise", "bias", "outlier", "acoustic", "atten", "snr", "depthsbl",
               "att", "est"]
GROUP_LABEL = {
    "noise":    ("§7 観測ノイズ", "sec7 noise"),
    "bias":     ("§8 バイアス/距離依存", "sec8 bias/growth"),
    "outlier":  ("§8 外れ値", "sec8 outliers"),
    "acoustic": ("§8.4/8.5 音速・同期", "sec8.4/8.5 sound/sync"),
    "atten":    ("§9 光減衰", "sec9 attenuation"),
    "snr":      ("§9 SNR/見失い", "sec9 SNR/dropout"),
    "depthsbl": ("§10 深度 / §13 SBL", "sec10 depth / sec13 SBL"),
    "att":      ("§14 波動揺", "sec14 wave sway"),
    "est":      ("§4.4 ロバスト損失", "sec4.4 robust loss"),
}

# spec: (key, ja, en, lo, hi, group, section, toml, angle, desc_ja, desc_en)
#   desc_* は「そのパラメータが何か」の一言説明 (UI のページ説明と config.toml の行コメントに出す)。
PARAM_SPECS = [
    ("sigma_dist", "σ_dist [m] (全手法)", "sigma_dist [m]", 0.0, 0.2, "noise", "noise", "sigma_dist", False,
     "音響測距の標準偏差 (全手法の距離ノイズ)", "stddev of acoustic range (all methods)"),
    ("sigma_az", "σ_az [deg] (光学)", "sigma_az [deg]", 0.0, 2.0, "noise", "noise", "sigma_az_deg", True,
     "カメラ方位角の測定ノイズ (光学)", "camera azimuth measurement noise (optical)"),
    ("sigma_el", "σ_el [deg] (光学)", "sigma_el [deg]", 0.0, 2.0, "noise", "noise", "sigma_el_deg", True,
     "カメラ仰角の測定ノイズ (光学)", "camera elevation measurement noise (optical)"),
    ("sigma_imu", "σ_imu [m/軸]", "sigma_imu [m]", 0.0, 0.2, "noise", "noise", "sigma_imu", False,
     "IMU 時刻間変位の軸あたりノイズ", "IMU per-axis displacement noise"),

    ("bias_dist", "距離バイアス [m]", "bias_dist [m]", -0.2, 0.2, "bias", "error_model", "bias_dist", False,
     "距離の系統オフセット (取付・校正残差)", "systematic range offset (mount/calib)"),
    ("bias_az", "方位バイアス [deg]", "bias_az [deg]", -1.0, 1.0, "bias", "error_model", "bias_az_deg", True,
     "方位角の系統ずれ (カメラ取付ミスアライン)", "systematic azimuth offset (misalign)"),
    ("bias_el", "仰角バイアス [deg]", "bias_el [deg]", -1.0, 1.0, "bias", "error_model", "bias_el_deg", True,
     "仰角の系統ずれ (カメラ取付ミスアライン)", "systematic elevation offset (misalign)"),
    ("range_growth", "角度の距離依存 [1/m]", "range_growth [1/m]", 0.0, 0.1, "bias", "error_model", "range_growth_per_m", False,
     "遠いほど角度ノイズが増す係数", "angle noise grows with range"),
    ("dist_growth", "距離の距離依存 [1/m]", "dist_growth [1/m]", 0.0, 0.1, "bias", "error_model", "dist_growth_per_m", False,
     "遠いほど距離ノイズが増す係数", "range noise grows with range"),

    ("outlier_rate", "外れ値率 [-]", "outlier_rate", 0.0, 0.3, "outlier", "error_model", "outlier_rate", False,
     "1観測あたり外れ値が出る確率 (見失い/マルチパス)", "per-sample outlier probability"),
    ("outlier_scale", "外れ値倍率 [×σ]", "outlier_scale", 1.0, 50.0, "outlier", "error_model", "outlier_scale", False,
     "外れ値の大きさ (有効σの倍率)", "outlier magnitude (× effective sigma)"),

    ("sound_speed_true", "真の音速 [m/s]", "c_true [m/s]", 1400.0, 1600.0, "acoustic", "acoustic", "sound_speed_true", False,
     "実際の水中音速 (真値)", "true underwater sound speed"),
    ("sound_speed_assumed", "仮定音速 [m/s]", "c_assumed [m/s]", 1400.0, 1600.0, "acoustic", "acoustic", "sound_speed_assumed", False,
     "推定側が仮定する音速 (真値とのズレが距離スケール誤差)", "assumed sound speed (mismatch=scale error)"),
    ("acoustic_latency", "音響遅延 [s]", "latency [s]", 0.0, 0.5, "acoustic", "sync", "acoustic_latency_s", False,
     "音響が光学より遅れる時間 (その間に子機が動く)", "acoustic lag behind optical"),

    ("clarity", "濁り c [1/m] (光学)", "clarity c [1/m]", 0.0, 1.5, "atten", "optical", "attenuation_c", False,
     "水の濁り (大きいほど遠方で精度低下・見失い)", "water turbidity (beam attenuation)"),
    ("range_ref", "基準距離 [m]", "range_ref [m]", 1.0, 30.0, "atten", "optical", "range_ref", False,
     "σ_ref / snr_ref を定義する基準距離", "reference range for sigma_ref/snr_ref"),
    ("sigma_ref", "基準σ_ang [deg]", "sigma_ref [deg]", 0.0, 2.0, "atten", "optical", "sigma_ref_deg", True,
     "基準距離での角度ノイズ (目標精度)", "angle noise at reference range"),
    ("sigma_floor", "σ_ang 下限 [deg]", "sigma_floor [deg]", 0.0, 1.0, "atten", "optical", "sigma_floor_deg", True,
     "ベストケース角度ノイズ (画素・校正限界)", "best-case angle noise floor"),

    ("snr_ref", "基準SNR [-]", "snr_ref", 1.0, 100.0, "snr", "optical", "snr_ref", False,
     "基準距離での SNR", "SNR at reference range"),
    ("snr_exponent", "SNR指数 [-]", "snr_exponent", 0.0, 2.0, "snr", "optical", "snr_exponent", False,
     "SNR の信号依存 (1=後方散乱律速 / 0.5=ショットノイズ)", "SNR signal dependence exponent"),
    ("snr_min", "最小SNR [-]", "snr_min", 0.0, 20.0, "snr", "optical", "snr_min", False,
     "検出に必要な最小 SNR (下回ると見失い増)", "min SNR for detection"),
    ("dropout_max", "最大見失い率 [-]", "dropout_max", 0.0, 1.0, "snr", "optical", "dropout_max", False,
     "SNR→0 での最大見失い確率", "max dropout prob as SNR->0"),
    ("dropout_jump", "見失い角飛び [deg]", "dropout_jump [deg]", 0.0, 90.0, "snr", "optical", "dropout_jump_deg", True,
     "見失い時の角度の飛び (誤検出相当の外れ値)", "angle jump on dropout (false detect)"),

    ("sigma_depth", "σ_depth [m]", "sigma_depth [m]", 0.0, 0.5, "depthsbl", "depth", "sigma_m", False,
     "圧力深度センサのノイズ", "pressure depth sensor noise"),
    ("depth_bias", "深度バイアス [m]", "depth_bias [m]", -0.5, 0.5, "depthsbl", "depth", "bias_m", False,
     "深度の系統バイアス (海面気圧・潮位ドリフト)", "depth systematic bias (tide/pressure)"),
    ("sbl_baseline", "SBL アレイ一辺 [m]", "SBL baseline [m]", 0.5, 8.0, "depthsbl", "sbl", "baseline", False,
     "親機トランスデューサ配置の一辺 (広いほど多辺測量が安定)", "SBL transducer array side length"),
    ("sbl_sigma_range", "SBL σ_range [m]", "SBL sigma_range [m]", 0.0, 0.2, "depthsbl", "sbl", "sigma_range", False,
     "SBL 各音響測距のノイズ", "per-anchor SBL range noise"),

    ("roll_amp", "roll振幅 [deg]", "roll amp [deg]", 0.0, 20.0, "att", "attitude", "roll_amp_deg", True,
     "波による横揺れ(roll)の振幅 (光学角度+SBLアレイに影響)", "wave roll amplitude (optical angle + SBL array)"),
    ("pitch_amp", "pitch振幅 [deg]", "pitch amp [deg]", 0.0, 20.0, "att", "attitude", "pitch_amp_deg", True,
     "波による縦揺れ(pitch)の振幅", "wave pitch amplitude"),
    ("yaw_amp", "yaw振幅 [deg]", "yaw amp [deg]", 0.0, 20.0, "att", "attitude", "yaw_amp_deg", True,
     "波による船首揺れ(yaw)の振幅 (方位角に直接効く)", "wave yaw amplitude (hits azimuth)"),
    ("roll_period", "roll周期 [s]", "roll period [s]", 1.0, 15.0, "att", "attitude", "roll_period_s", False,
     "roll 揺れの主要周期 (波周期)", "roll sway main period"),
    ("pitch_period", "pitch周期 [s]", "pitch period [s]", 1.0, 15.0, "att", "attitude", "pitch_period_s", False,
     "pitch 揺れの主要周期", "pitch sway main period"),
    ("yaw_period", "yaw周期 [s]", "yaw period [s]", 1.0, 20.0, "att", "attitude", "yaw_period_s", False,
     "yaw 揺れの主要周期 (係留でゆっくり)", "yaw sway main period"),

    ("f_scale", "損失 f_scale [×σ]", "f_scale", 0.1, 5.0, "est", "estimator", "f_scale", False,
     "ロバスト損失の内れ値しきい値 (σ単位)", "robust-loss inlier threshold (in sigma)"),
]
SPEC_BY_KEY = {sp[0]: dict(key=sp[0], ja=sp[1], en=sp[2], lo=sp[3], hi=sp[4],
                           group=sp[5], section=sp[6], toml=sp[7], angle=sp[8],
                           desc_ja=sp[9], desc_en=sp[10])
               for sp in PARAM_SPECS}

# config.toml 出力の行コメント (section, key) -> 一言説明。各パラメータ + enable/loss。
_TOML_COMMENTS = {(sp[6], sp[7]): sp[9] for sp in PARAM_SPECS}
_TOML_COMMENTS.update({
    ("error_model", "enable"): "現実誤差§8 を有効化 (UI/CLI の ON/OFF)",
    ("optical", "enable"): "光減衰§9 を有効化 (UI/CLI の ON/OFF)",
    ("depth", "enable"): "深度センサ§10 を有効化 (UI/CLI の ON/OFF)",
    ("estimator", "loss"): "ロバスト損失の種類 (§4.4)",
    ("attitude", "as_error"): "波動揺§14 を光学角度の誤差として適用 (UI/CLI の ON/OFF)",
    ("attitude", "imu_correct"): "波動揺適用時に IMU 相補フィルタで姿勢補正するか (false=naive)",
})
GROUP_PARAMS = {g: [sp[0] for sp in PARAM_SPECS if sp[5] == g] for g in GROUP_ORDER}
MAX_SLOTS = max(len(v) for v in GROUP_PARAMS.values())   # 1 ページの最大スライダ数

# config.toml から取り込む初期値 (角度は [rad]→[deg] に戻す)。誤差モデルの変更が反映される。
DEFAULTS = {
    "sigma_dist": float(SIGMA_DIST),
    "sigma_az": float(np.rad2deg(SIGMA_AZ)),
    "sigma_el": float(np.rad2deg(SIGMA_EL)),
    "sigma_imu": float(SIGMA_IMU),
    "bias_dist": float(ERROR_MODEL["bias"][0]),
    "bias_az": float(np.rad2deg(ERROR_MODEL["bias"][1])),
    "bias_el": float(np.rad2deg(ERROR_MODEL["bias"][2])),
    "range_growth": float(ERROR_MODEL["range_growth_per_m"]),
    "dist_growth": float(ERROR_MODEL["dist_growth_per_m"]),
    "outlier_rate": float(ERROR_MODEL["outlier_rate"]),
    "outlier_scale": float(ERROR_MODEL["outlier_scale"]),
    "sound_speed_true": float(ERROR_MODEL["sound_speed_true"]),
    "sound_speed_assumed": float(ERROR_MODEL["sound_speed_assumed"]),
    "acoustic_latency": float(ERROR_MODEL["acoustic_latency_s"]),
    "clarity": float(OPTICAL_MODEL["attenuation_c"]),
    "range_ref": float(OPTICAL_MODEL["range_ref"]),
    "sigma_ref": float(np.rad2deg(OPTICAL_MODEL["sigma_ref"])),
    "sigma_floor": float(np.rad2deg(OPTICAL_MODEL["sigma_floor"])),
    "snr_ref": float(OPTICAL_MODEL["snr_ref"]),
    "snr_exponent": float(OPTICAL_MODEL["snr_exponent"]),
    "snr_min": float(OPTICAL_MODEL["snr_min"]),
    "dropout_max": float(OPTICAL_MODEL["dropout_max"]),
    "dropout_jump": float(np.rad2deg(OPTICAL_MODEL["dropout_jump"])),
    "sigma_depth": float(SIGMA_DEPTH),
    "depth_bias": float(DEPTH_BIAS),
    "sbl_baseline": float(SBL_BASELINE),
    "sbl_sigma_range": float(SBL_SIGMA_RANGE),
    "roll_amp": float(np.rad2deg(ATT_ROLL_AMP)),
    "pitch_amp": float(np.rad2deg(ATT_PITCH_AMP)),
    "yaw_amp": float(np.rad2deg(ATT_YAW_AMP)),
    "roll_period": float(ATT_ROLL_PERIOD),
    "pitch_period": float(ATT_PITCH_PERIOD),
    "yaw_period": float(ATT_YAW_PERIOD),
    "f_scale": float(EST_F_SCALE),
}


def initial_state():
    """config.toml を初期値とする state 辞書 (全パラメータ + ON/OFF + 損失 + 手法)。"""
    st = {k: float(v) for k, v in DEFAULTS.items()}
    st.update(use_depth=bool(DEPTH_ENABLE), use_error=bool(ERROR_MODEL_ENABLE),
              use_atten=bool(OPTICAL_ENABLE),
              use_attitude=bool(ATT_AS_ERROR), att_correct=bool(ATT_IMU_CORRECT),
              loss=(EST_LOSS if EST_LOSS in LOSS_CHOICES else "linear"),
              m_optical=True, m_sbl=True, m_no_optical=True)
    return st


def _att_wave(state):
    """state の波動揺パラメータ (deg/s) を wave_attitude_sequence 用の辞書 (rad/s) にする。"""
    return {"roll_amp": np.deg2rad(state["roll_amp"]),
            "pitch_amp": np.deg2rad(state["pitch_amp"]),
            "yaw_amp": np.deg2rad(state["yaw_amp"]),
            "roll_period": state["roll_period"], "pitch_period": state["pitch_period"],
            "yaw_period": state["yaw_period"], "yaw_mean": ATT_YAW_MEAN}


def _print_params():
    """調整可能パラメータの一覧 (群・キー・説明・初期値) を表示する (--list-params)。"""
    print("=== 調整可能パラメータ (--set KEY=VALUE で上書き / config.toml は不変) ===")
    for g in GROUP_ORDER:
        print("\n[%s]" % L(*GROUP_LABEL[g]))
        for key in GROUP_PARAMS[g]:
            sp = SPEC_BY_KEY[key]
            print("  %-20s 初期=%-9.4g  %s" % (key, DEFAULTS[key], L(sp["desc_ja"], sp["desc_en"])))
    print("\n[ON/OFF]  --depth/--error/--attenuation (初期=config の enable)")
    print("[loss]    --loss {%s} (§4.4 ロバスト損失)" % "/".join(LOSS_CHOICES))


def _resolve_workers(requested=None):
    """並列ワーカー数を決める。requested=None なら WORKERS、それも None なら CPU 数。"""
    n = requested if requested is not None else WORKERS
    if n is None:
        n = os.cpu_count() or 1
    return max(1, int(n))


# =====================================================================
# state → 各層が必要とするモデル辞書/タプルへの変換 (角度は deg→rad)
# =====================================================================
def _sig_gen(state):
    """観測生成に使う (σ_dist, σ_az, σ_el) [m, rad, rad]。"""
    return (state["sigma_dist"], np.deg2rad(state["sigma_az"]), np.deg2rad(state["sigma_el"]))


def _opt_model(state):
    """§9 光減衰モデル辞書 (sensors の optical_* に渡す。角度は rad)。"""
    return {"attenuation_c": state["clarity"], "range_ref": state["range_ref"],
            "sigma_ref": np.deg2rad(state["sigma_ref"]),
            "sigma_floor": np.deg2rad(state["sigma_floor"]),
            "snr_ref": state["snr_ref"], "snr_exponent": state["snr_exponent"],
            "snr_min": state["snr_min"], "dropout_max": state["dropout_max"],
            "dropout_jump": np.deg2rad(state["dropout_jump"])}


def _error_kw(state):
    """§8 現実誤差を realistic 観測 (光学/光学なし) に渡すキーワード (角度バイアスは rad)。"""
    return {"bias": (state["bias_dist"], np.deg2rad(state["bias_az"]),
                     np.deg2rad(state["bias_el"])),
            "range_growth_per_m": state["range_growth"],
            "dist_growth_per_m": state["dist_growth"],
            "outlier_rate": state["outlier_rate"], "outlier_scale": state["outlier_scale"],
            "sound_speed_true": state["sound_speed_true"],
            "sound_speed_assumed": state["sound_speed_assumed"],
            "acoustic_latency_s": state["acoustic_latency"]}


def _sbl_error_kw(state):
    """SBL 測距に効く §8 音響誤差 (距離系のみ。角度・同期は SBL に無い)。"""
    return {"sound_speed_true": state["sound_speed_true"],
            "sound_speed_assumed": state["sound_speed_assumed"],
            "bias_dist": state["bias_dist"], "dist_growth_per_m": state["dist_growth"],
            "outlier_rate": state["outlier_rate"], "outlier_scale": state["outlier_scale"]}


def _est_loss(state):
    """推定の損失関数。state["loss"] を尊重し、誤差ONで純L2 のままなら外れ値対策に huber へ昇格。"""
    loss = state.get("loss", "linear")
    if loss != "linear":
        return loss
    return "huber" if state["use_error"] else "linear"


def _traj(depth):
    """near-nadir サーベイ軌道 (config [survey])。全手法で共有。"""
    return double_lawnmower_trajectory(area=SURVEY_AREA, depth=-float(depth),
                                       n_legs=N_LEGS, pts_per_leg=PTS_PER_LEG,
                                       origin=SURVEY_ORIGIN)


def _anchors(baseline):
    b = baseline / 2.0
    return np.array([[b, b, 0.0], [b, -b, 0.0], [-b, b, 0.0], [-b, -b, 0.0]])


def _depth_kw(state, traj, seed):
    """深度観測 (§10) を作って estimate_trajectory に渡す kwargs を返す (use_depth 時のみ)。"""
    dep = simulate_depth_sequence(traj, state["sigma_depth"], seed=substream_seed(seed, 2),
                                  bias=state["depth_bias"])
    return dict(z_depth_seq=dep, sigma_depth=state["sigma_depth"])


# =====================================================================
# 各手法の RMSE 評価 (state のみ入力。truth は評価時のみ参照 = MBD 分離)
# =====================================================================
def _optical_est(state, seed):
    """光学アームの (真値軌道, 推定軌道) を返す (RMSE 計算と算出方法の図示で共用)。"""
    traj = _traj(state["depth"])
    model = _opt_model(state) if state["use_atten"] else None
    err = _error_kw(state) if state["use_error"] else {}
    sig_gen = _sig_gen(state)
    if model is not None or err:
        z = simulate_observation_sequence_realistic(
            traj, sig_gen, seed=substream_seed(seed, 0), p_parent=P_PARENT, optical_model=model, **err)
    else:
        z = simulate_observation_sequence(traj, sig_gen, seed=substream_seed(seed, 0), p_parent=P_PARENT)
    if state.get("use_attitude"):          # §14 波による親機動揺を角度観測に重ねる (距離は不感)
        z = apply_attitude_error(z, seed=substream_seed(seed, 4), enable=True, imu_correct=state["att_correct"],
                                 wave=_att_wave(state), dt=ATT_DT,
                                 filter_alpha=ATT_FILTER_ALPHA, **ATT_IMU_KW)
    imu = simulate_imu_displacements(traj, state["sigma_imu"], seed=substream_seed(seed, 1))
    if model is not None:        # §9 校正σ (適応重み, 各シナリオと統一)
        s_ang = optical_angular_sigma(float(np.linalg.norm(traj.mean(axis=0))), model)
        weight = (state["sigma_dist"], s_ang, s_ang)
    else:
        weight = sig_gen
    kw = _depth_kw(state, traj, seed) if state["use_depth"] else {}
    est = estimate_trajectory(z, weight, imu_deltas=imu, sigma_imu=state["sigma_imu"],
                              p_parent=P_PARENT, loss=_est_loss(state),
                              f_scale=state["f_scale"], **kw)
    return traj, est


def _rmse_optical(state, seed):
    traj, est = _optical_est(state, seed)
    return rmse_xyz(traj, est)["total"] * 1000


def _rmse_sbl(state, seed):
    traj = _traj(state["depth"])
    anchors = _anchors(state["sbl_baseline"])
    err = _sbl_error_kw(state) if state["use_error"] else {}
    # §14 親機の波動揺: トランスデューサアレイは親機ピボットからオフセットして付くので、波で
    # 親機が揺れるとアレイが回り各レンジが変わる (§13.5)。真値レンジは回ったアンカーで生成し、
    # 推定は att_correct=True なら IMU 相補フィルタ姿勢でアンカーを回す / False なら公称のまま。
    # (光学角度と対称。単一距離=ピボット上の1点は回転不変なので光学なしアームは不感。)
    anchors_true, anchors_est = anchors, anchors
    if state.get("use_attitude"):
        anchors_true, anchors_est = sbl_attitude_anchors(
            anchors, len(traj), seed=substream_seed(seed, 3), enable=True,
            imu_correct=state["att_correct"], wave=_att_wave(state), dt=ATT_DT,
            p_parent=P_PARENT, filter_alpha=ATT_FILTER_ALPHA, **ATT_IMU_KW)
    rng = simulate_sbl_range_sequence(traj, anchors_true, state["sbl_sigma_range"],
                                      seed=substream_seed(seed, 0), **err)
    imu = simulate_imu_displacements(traj, state["sigma_imu"], seed=substream_seed(seed, 1))
    dep = simulate_depth_sequence(traj, state["sigma_depth"], seed=substream_seed(seed, 2),
                                  bias=state["depth_bias"])
    est = estimate_trajectory_sbl(rng, anchors_est, state["sbl_sigma_range"], imu,
                                  state["sigma_imu"], dep, state["sigma_depth"],
                                  p_parent=P_PARENT, loss=_est_loss(state),
                                  f_scale=state["f_scale"])
    return rmse_xyz(traj, est)["total"] * 1000


def _rmse_no_optical(state, seed):
    traj = _traj(state["depth"])
    sig_gen = _sig_gen(state)             # 角度は使わない (距離のみ) が生成は full obs
    if state["use_error"]:
        z = simulate_observation_sequence_realistic(traj, sig_gen, seed=substream_seed(seed, 0),
                                                    p_parent=P_PARENT, **_error_kw(state))
    else:
        z = simulate_observation_sequence(traj, sig_gen, seed=substream_seed(seed, 0), p_parent=P_PARENT)
    imu = simulate_imu_displacements(traj, state["sigma_imu"], seed=substream_seed(seed, 1))
    dep = simulate_depth_sequence(traj, state["sigma_depth"], seed=substream_seed(seed, 2),
                                  bias=state["depth_bias"])
    est = estimate_trajectory_acoustic_inertial(
        z[:, 0], state["sigma_dist"], imu, state["sigma_imu"], dep, state["sigma_depth"],
        p_parent=P_PARENT, n_azimuth_starts=NO_OPT_AZ,
        loss=_est_loss(state), f_scale=state["f_scale"])
    return rmse_xyz(traj, est)["total"] * 1000


_RMSE_FN = {"optical": _rmse_optical, "sbl": _rmse_sbl, "no_optical": _rmse_no_optical}


def _eval_task(task):
    """1 つの (手法, state, seed) を評価して RMSE [mm] を返す (並列ワーカーの実行単位)。

    トップレベル関数なので ProcessPoolExecutor で pickle して別プロセスへ渡せる。
    state は float/bool/str のみ、seed は int で完全に再現可能 (seed0+s を呼び出し側で確定)。
    """
    key, state, seed = task
    return _RMSE_FN[key](state, seed)


def _run_tasks(tasks, workers):
    """評価タスク列を順序を保って実行し、結果リストを返す。

    workers<=1 / タスク1個 / 並列起動失敗時は逐次実行にフォールバックする (再現値は同一)。
    各タスクは独立した最小二乗推定で CPU バウンドなので、プロセス並列がそのまま効く。
    """
    if not tasks:
        return []
    n_workers = min(_resolve_workers(workers), len(tasks))
    if n_workers <= 1 or len(tasks) == 1:
        return [_eval_task(t) for t in tasks]
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(_eval_task, tasks))
    except Exception as e:        # 並列が使えない環境では逐次に落とす (結果は同じ)
        print(f"[explore] 並列実行に失敗 -> 逐次実行にフォールバックします: {e}")
        return [_eval_task(t) for t in tasks]


def compute_curves(state, depths, n_seeds=N_SEEDS, seed0=SEED, workers=None):
    """有効手法ごとに、各水深での total RMSE [mm] のリストを順序付き dict で返す。

    水深を横軸に掃引する: depths の各点で state の depth を差し替えて評価する。
    (手法 x 水深 x seed) の全評価を 1 つのタスク列にまとめ、CPU 並列で実行してから
    seed 平均を取る (逐次と同じ値・再現性。workers で並列度を制御)。
    """
    active = [k for k in METHOD_KEYS if state["m_" + k]]
    if not active:
        return {}
    tasks, index = [], []     # index[i] = (手法, 水深インデックス) で結果を振り分ける
    for key in active:
        for di, d in enumerate(depths):
            st = dict(state)
            st["depth"] = float(d)
            for s in range(n_seeds):
                tasks.append((key, st, substream_seed(seed0, s)))   # 独立試行 (§15.2)
                index.append((key, di))
    vals = _run_tasks(tasks, workers)
    acc = {key: [[] for _ in depths] for key in active}
    for (key, di), v in zip(index, vals):
        acc[key][di].append(v)
    return {key: [float(np.mean(acc[key][di])) for di in range(len(depths))]
            for key in active}


# =====================================================================
# 実効設定の config.toml 出力 (UI/CLI 上書き後の最終パラメータをスナップショット)
# =====================================================================
_SECTION_ORDER = ["coords", "noise", "rates", "error_model", "acoustic", "sync",
                  "switch", "sbl", "depth", "estimator", "optical", "attitude",
                  "deepwater", "truth", "cube", "stereo", "survey", "trajectory",
                  "demo_trajectory", "montecarlo", "mapping", "sensitivity", "spec",
                  "visualization"]


def _toml_value(v):
    """Python 値を TOML リテラル文字列にする (スカラ + 数値配列のみ。config.toml は平坦)。"""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # 10 桁有効数字に丸めて deg→rad→deg 等の浮動小数ノイズを除去
        # (29.999999999999996 → 30.0)。9.80665 等の正当な精度は保つ。
        return repr(float("%.10g" % v))   # 例: 0.3 / 1500.0 / 30.0
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return _toml_value(float(v))           # numpy スカラ等は float へ


def _toml_dumps(cfg, header_lines=()):
    """セクション辞書 cfg を config.toml テキストに整形する (header_lines はコメント)。"""
    out = list(header_lines)
    if out:
        out.append("")
    done = set()

    def _emit(sec):
        data = cfg.get(sec)
        if not isinstance(data, dict):
            return
        out.append(f"[{sec}]")
        for k, val in data.items():
            try:
                line = f"{k} = {_toml_value(val)}"
            except Exception:
                continue                   # 表現できない値はスキップ (防御的)
            desc = _TOML_COMMENTS.get((sec, k))
            out.append(line + (f"  # {desc}" if desc else ""))
        out.append("")
        done.add(sec)

    for sec in _SECTION_ORDER:
        if sec in cfg:
            _emit(sec)
    for sec in cfg:                        # 想定外セクションも漏らさず出力
        if sec not in done:
            _emit(sec)
    return "\n".join(out).rstrip() + "\n"


def _effective_config(state):
    """元 config.toml を土台に、explore が管理する全パラメータを state で上書きした辞書。"""
    cfg = copy.deepcopy(dict(CONFIG_RAW)) if CONFIG_RAW else {}

    def setk(sec, key, val):
        cfg.setdefault(sec, {})[key] = val

    for sp in PARAM_SPECS:               # §7-§13/§4.4 の全数値 (角度は deg のまま書く)
        setk(sp[6], sp[7], float(state[sp[0]]))
    setk("error_model", "enable", bool(state["use_error"]))   # ON/OFF も実効値で
    setk("optical", "enable", bool(state["use_atten"]))
    setk("depth", "enable", bool(state["use_depth"]))
    setk("estimator", "loss", str(state["loss"]))
    setk("attitude", "as_error", bool(state["use_attitude"]))     # §14 波動揺を誤差として適用
    setk("attitude", "imu_correct", bool(state["att_correct"]))
    return cfg


def _write_effective_config(state, depths, n_seeds, seed, outdir):
    """results/explore/config.toml に実効設定を書き出す (元 config.toml は変更しない)。"""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    methods = ", ".join(k for k in METHOD_KEYS if state["m_" + k]) or "(なし)"
    header = [
        "# =====================================================================",
        "# 自動生成: run_explore.py の最終実行設定スナップショット (config.toml 形式)",
        "# =====================================================================",
        "# これは UI/CLI で上書きした「実効パラメータ」を config.toml として書き出したもの。",
        "# 元のリポジトリ直下 config.toml は変更していない。この内容で実行を再現できる",
        "# (この値で本番計算したい場合はリポジトリ直下 config.toml にコピーする)。",
        f"# 生成時刻 (UTC): {stamp}",
        f"# 掃引した水深グリッド [m]: {', '.join('%g' % d for d in depths)}",
        f"# モンテカルロ: seed={seed}, n_seeds={n_seeds}",
        f"# 有効手法: {methods}",
        f"# 角度は [deg] (config と同じ *_deg キー)。enable は UI/CLI の ON/OFF を反映。",
    ]
    text = _toml_dumps(_effective_config(state), header)
    path = os.path.join(outdir, "config.toml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# =====================================================================
# 表示・出力
# =====================================================================
def _cond_text(state):
    att = ("ON" + (L("補正", "corr") if state["att_correct"] else L("naive", "naive"))
           if state["use_attitude"] else "OFF")
    opts = [L("深度", "depth") + ("ON" if state["use_depth"] else "OFF"),
            L("§8誤差", "err") + ("ON" if state["use_error"] else "OFF"),
            L("§9減衰", "atten") + ("ON" if state["use_atten"] else "OFF"),
            L("§14波", "wave") + att]
    return (L("c%.2f σ_az%.2f° σ_d%.1fcm SBL%.1fm loss=%s | %s (詳細→config.toml)",
              "c%.2f saz%.2f sd%.1fcm SBL%.1fm loss=%s | %s (see config.toml)")
            % (state["clarity"], state["sigma_az"], state["sigma_dist"] * 100,
               state["sbl_baseline"], state["loss"], " ".join(opts)))


def _draw_curves(ax, depths, curves):
    ax.clear()
    for key, ys in curves.items():
        ax.plot(depths, ys, "o-", color=METHOD_COLOR[key], label=METHOD_LABEL[key])
    ax.set_xlabel(L("水深 [m]", "depth [m]"))
    ax.set_ylabel("RMSE total [mm]")
    if curves:
        ax.set_yscale("log")
        ax.legend(fontsize=9, loc="upper left")
    ax.set_title(L("RMSE vs 水深 (手法別・低いほど良い)", "RMSE vs depth (lower better)"))
    ax.grid(alpha=0.3, which="both")


def _method_figure(state, depths, n_seeds, seed, outdir):
    """RMSE 算出方法の説明図。左=光学の真値 vs 推定で残差を可視化 / 右=手順と RMSE 式。"""
    ex = dict(state)
    ex["depth"] = float(depths[len(depths) // 2])      # 代表水深 (中央)
    traj, est = _optical_est(ex, seed)
    r = rmse_xyz(traj, est)["total"] * 1000

    fig = plt.figure(figsize=(13.0, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="red", lw=1.8,
            label=L("真値軌道 (truth)", "truth"))
    ax.scatter(est[:, 0], est[:, 1], est[:, 2], c="tab:blue", s=24,
               label=L("推定 (estimate)", "estimate"))
    for k in range(len(traj)):
        ax.plot([traj[k, 0], est[k, 0]], [traj[k, 1], est[k, 1]],
                [traj[k, 2], est[k, 2]], color="gray", lw=0.6)
    ax.set_title(L("例: 光学 深さ%gm  RMSE %.0fmm\n灰線=推定と真値の差(残差)" % (ex["depth"], r),
                   "optical @ %gm  RMSE %.0fmm" % (ex["depth"], r)), fontsize=10)
    ax.set_xlabel("X[m]"); ax.set_ylabel("Y[m]"); ax.set_zlabel("Z[m]")
    ax.legend(fontsize=8, loc="upper left")

    axt = fig.add_subplot(1, 2, 2)
    axt.axis("off")
    grid = ", ".join("%g" % d for d in depths)
    steps_ja = (
        "RMSE 算出方法 (各手法 x 各水深)\n\n"
        "1) 真値: near-nadir 芝刈り軌道\n"
        "   (config [survey], z = -depth)\n"
        "2) 観測生成 (sensors):\n"
        "   §7 零平均ノイズ\n"
        "   + §8 現実誤差 (任意, [error_model])\n"
        "   + §9 光減衰 (光学のみ, 任意, [optical])\n"
        "3) 推定 (estimator, 観測のみ / truth 非参照):\n"
        "   ・光学    = 角度 + 距離 + IMU (+深度)\n"
        "   ・SBL     = 4距離(多辺測量) + IMU + 深度\n"
        "   ・光学なし = 距離 + IMU + 深度\n"
        "4) 評価 (evaluation):\n"
        "   RMSE_total = sqrt( (1/N) Σ_k || p_hat,k - p_k ||^2 )\n"
        "5) seed を %d 回平均し、水深グリッドを掃引\n\n"
        "水深グリッド [m]: %s" % (n_seeds, grid))
    steps_en = (
        "RMSE computation (per method x depth)\n\n"
        "1) truth: near-nadir lawnmower path ([survey])\n"
        "2) observe (sensors): sec7 noise\n"
        "   + sec8 errors + sec9 attenuation (optical)\n"
        "3) estimate (obs only, no truth):\n"
        "   optical = angle + range + IMU (+depth)\n"
        "   SBL = 4 ranges (multilateration) + IMU + depth\n"
        "   no-optical = range + IMU + depth\n"
        "4) RMSE_total = sqrt( (1/N) sum_k ||p_hat - p||^2 )\n"
        "5) average %d seeds, sweep the depth grid\n\n"
        "depth grid [m]: %s" % (n_seeds, grid))
    axt.text(0.0, 1.0, L(steps_ja, steps_en), va="top", ha="left", fontsize=11)

    fig.suptitle(L("explore の RMSE 算出方法", "How explore computes RMSE"))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rmse_method.png"), bbox_inches="tight")
    plt.close(fig)


def _write_report_explore(state, depths, curves, n_seeds, seed):
    """results/explore/ に図 (explore.png, rmse_method.png)・実効 config.toml・README を書く。"""
    outdir = scenario_dir("explore")
    # (1) RMSE vs 水深 曲線
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _draw_curves(ax, depths, curves)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "explore.png"), bbox_inches="tight")
    plt.close(fig)
    # (2) 算出方法の説明図
    _method_figure(state, depths, n_seeds, seed, outdir)
    # (3) 実効設定の config.toml スナップショット (UI/CLI 上書き後の最終値)
    _write_effective_config(state, depths, n_seeds, seed, outdir)
    # (4) README (代表結果 = 中央水深での各手法 RMSE)
    j = len(depths) // 2
    results = {METHOD_LABEL[k]: "%.0f mm (深さ%gm)" % (curves[k][j], depths[j])
               for k in curves} if curves else {}
    write_report(
        "explore", "対話的パラメータ探索 (光学/SBL/光学なし の RMSE vs 水深)",
        "config.toml を編集せず、スライダ/チェックで**全誤差モデルを上書き**し [計算] ボタンで 3 手法の\n"
        "「RMSE vs 水深」を比較する探索ツール。パラメータは種類ごとのページ (§7/§8/§8.4-8.5/§9/§10/§13/§14/§4.4)\n"
        "に整理し、各ページのスライダで操作する。RMSE は次の手順で算出する:\n"
        "(1) 真値 = near-nadir 芝刈り軌道 (config [survey], z=-depth)、(2) 観測 = §7 零平均ノイズ\n"
        "(+§8 現実誤差 / +§9 光減衰 は任意)、(3) 観測のみから各手法で軌道推定 (estimator, truth 非参照)、\n"
        "(4) RMSE_total = sqrt( (1/N)Σ||p̂_k − p_k||² )、(5) seed を平均し水深グリッドを掃引 (CPU 並列)。\n"
        "**この実行で実際に使った全パラメータは同フォルダの `config.toml` に出力**している (下記)。",
        not_reflected=[
            ("UI/CLI で上書きする全誤差パラメータ (config.toml は不変)",
             "本ツールは §7 ノイズ / §8 現実誤差・音速・同期 / §9 光減衰 / §10 深度 / §13 SBL / "
             "§14 波動揺 / §4.4 ロバスト損失の**全パラメータ**をスライダ/`--set`/各 CLI で上書きできる。"
             "config.toml の値は**初期値**として読み込むだけで、上書き後の最終値は同梱 `config.toml` (実効設定) に出力する。"),
            ("ON/OFF (深度§10 / 現実誤差§8 / 光減衰§9 / 波動揺§14) は config の enable が初期値",
             "各 `enable` ([depth]/[error_model]/[optical]/[attitude].as_error) を初期 ON/OFF とし、"
             "UI チェックや `--depth/--error/--attenuation/--attitude` で上書きする。誤差 ON 時のみ "
             "§8/§8.4/§8.5 が観測に乗る。波動揺§14 ON 時は親機の波動揺が**光学角度と SBL アンカー"
             "アレイ**に乗り (§13.5。単一距離=ピボット上の1点は回転不変なので光学なしは不感)、"
             "`--att-correct/--no-att-correct` で IMU 姿勢補正 (baseline 付近へ回復) / naive (誤差が残る) を選ぶ。"),
            ("`[stereo]`/`[cube]`/`[trajectory]`/`[attitude]`/`[switch]`/`[spec]` 等",
             "本ツールは測位3手法の比較のみ。ステレオ計測・キューブ・親機姿勢・自動切替・設計逆算は扱わない"
             "(実効 config.toml にはこれらのセクションも参考のため元の値のまま出力する)。"),
        ],
        outputs=[("explore.png", "RMSE vs 水深 (光学/SBL/光学なし の比較曲線)"),
                 ("rmse_method.png", "RMSE 算出方法の説明図 (残差の可視化 + 手順・式)"),
                 ("config.toml", "この実行の実効設定 (UI/CLI 上書き後の最終パラメータ) を config.toml 形式で出力")],
        results=results,
        meta={"seed": seed, "n_seeds": n_seeds, "loss": state["loss"],
              "depths_m": ",".join("%g" % d for d in depths)},
        math_spec="§4.4,§5,§7-§14")


def _one_shot(state, depths, n_seeds, seed, save=True, workers=None):
    curves = compute_curves(state, depths, n_seeds, seed, workers=workers)
    print("=== 測位手法 RMSE vs 水深 (一発計算) ===")
    print("  " + _cond_text(state) + f" / MC={n_seeds}seeds")
    if not curves:
        print("  (有効な手法がありません)")
        return curves
    print("  水深[m]      " + "".join("%8.1f" % d for d in depths))
    for k, ys in curves.items():
        print(f"  {METHOD_LABEL[k]:10s}" + "".join("%8.0f" % y for y in ys))
    if save:
        _write_report_explore(state, depths, curves, n_seeds, seed)
        print(f"  -> 出力: {scenario_dir('explore')} "
              "(explore.png / rmse_method.png / config.toml / README.md)")
    return curves


def _interactive(state, depths, n_seeds, seed, workers=None):
    fig = plt.figure(figsize=(13.5, 8.0))
    ax = fig.add_axes([0.30, 0.60, 0.66, 0.34])      # RMSE 曲線 (上)
    title = fig.suptitle("", fontsize=10)
    prog = {"on": False}      # set_val 等のプログラム更新中は state を書かない/未計算化しない

    def _mark_stale():
        if prog["on"]:
            return
        title.set_text(L("[未計算] パラメータ変更 → 右下の [計算] ボタンを押してください",
                         "[stale] change params, then press [Compute]"))
        fig.canvas.draw_idle()

    def _compute(_event=None):
        """計算ボタンで呼ぶ重い処理。ここでだけ水深を掃引して全手法を再計算する (CPU 並列)。"""
        title.set_text(L("計算中… (CPU 並列。しばらくお待ちください)", "computing… (parallel)"))
        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()    # 「計算中」表示を先に出す
        except Exception:
            pass
        curves = compute_curves(state, depths, n_seeds, seed, workers=workers)
        _draw_curves(ax, depths, curves)
        _write_report_explore(state, depths, curves, n_seeds, seed)   # 実効 config.toml も更新
        title.set_text(_cond_text(state))
        fig.canvas.draw_idle()

    # --- パラメータ群ラジオ (ページ切替。左上) ----------------------------------
    cat_ax = fig.add_axes([0.02, 0.40, 0.16, 0.40])
    cat_ax.set_title(L("パラメータ群", "param group"), fontsize=9)
    cat_labels = [L(*GROUP_LABEL[g]) for g in GROUP_ORDER]
    cat_radio = RadioButtons(cat_ax, cat_labels, active=0)

    # --- スライダプール (固定 MAX_SLOTS 本。ページに応じて中身を差し替える) -------
    slot_ax = [fig.add_axes([0.32, 0.52 - i * 0.046, 0.40, 0.028]) for i in range(MAX_SLOTS)]
    slot_sliders, slot_key = [], [None] * MAX_SLOTS
    for i, a in enumerate(slot_ax):
        s = Slider(a, "", 0.0, 1.0, valinit=0.5)

        def _mk(idx):
            def _cb(val):
                k = slot_key[idx]
                if k is not None and not prog["on"]:
                    state[k] = float(val)
                    _mark_stale()
            return _cb
        s.on_changed(_mk(i))
        slot_sliders.append(s)

    # 各スライダの一言説明 (ページ切替で更新)。スライダ下に置く。
    desc_text = fig.text(0.205, 0.255, "", fontsize=8.5, va="top", ha="left",
                         family=(_JP or "sans-serif"),
                         bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#cccccc"))

    def _apply_group(gid):
        """選択された群のパラメータを MAX_SLOTS 本のスロットに割り当てる (余りは隠す)。

        各スライダの一言説明も desc_text に並べ、何を動かしているかが分かるようにする。
        """
        keys = GROUP_PARAMS[gid]
        prog["on"] = True
        lines = [L("このページのパラメータ (一言説明):", "parameters on this page:")]
        for i in range(MAX_SLOTS):
            a, s = slot_ax[i], slot_sliders[i]
            if i < len(keys):
                sp = SPEC_BY_KEY[keys[i]]
                val = float(state[sp["key"]])
                lo, hi = min(sp["lo"], val), max(sp["hi"], val)
                if hi <= lo:
                    hi = lo + 1.0
                s.valmin, s.valmax = lo, hi
                a.set_xlim(lo, hi)
                s.label.set_text(L(sp["ja"], sp["en"]))
                slot_key[i] = sp["key"]
                s.set_val(val)
                a.set_visible(True)
                lines.append("・%s : %s" % (L(sp["ja"], sp["en"]), L(sp["desc_ja"], sp["desc_en"])))
            else:
                slot_key[i] = None
                a.set_visible(False)
        desc_text.set_text("\n".join(lines))
        prog["on"] = False
        fig.canvas.draw_idle()

    def _on_cat(label):
        _apply_group(GROUP_ORDER[cat_labels.index(label)])
    cat_radio.on_clicked(_on_cat)

    # --- 損失関数ラジオ (§4.4。左下) -------------------------------------------
    loss_ax = fig.add_axes([0.02, 0.10, 0.16, 0.24])
    loss_ax.set_title(L("損失 (§4.4)", "loss (sec4.4)"), fontsize=9)
    loss_active = LOSS_CHOICES.index(state["loss"]) if state["loss"] in LOSS_CHOICES else 0
    loss_radio = RadioButtons(loss_ax, LOSS_CHOICES, active=loss_active)

    def _on_loss(label):
        state["loss"] = label
        _mark_stale()
    loss_radio.on_clicked(_on_loss)

    # --- 手法 ON/OFF (右) -------------------------------------------------------
    m_ax = fig.add_axes([0.80, 0.40, 0.18, 0.16])
    m_ax.set_title(L("手法 (重いと感じたら切る)", "methods"), fontsize=9)
    m_labels = [METHOD_LABEL[k] for k in METHOD_KEYS]
    m_check = CheckButtons(m_ax, m_labels, [state["m_" + k] for k in METHOD_KEYS])

    def _on_method(label):
        key = METHOD_KEYS[m_labels.index(label)]
        state["m_" + key] = not state["m_" + key]
        _mark_stale()
    m_check.on_clicked(_on_method)

    # --- ON/OFF オプション (右) -------------------------------------------------
    o_ax = fig.add_axes([0.80, 0.18, 0.18, 0.20])
    o_ax.set_title(L("ON/OFF", "toggles"), fontsize=9)
    o_labels = [L("深度§10(光学)", "depth sensor"), L("現実誤差§8", "error model"),
                L("光減衰§9(光学)", "attenuation"), L("波動揺§14(光学)", "wave sway"),
                L("  └姿勢IMU補正", "  IMU correct")]
    o_keys = ["use_depth", "use_error", "use_atten", "use_attitude", "att_correct"]
    o_check = CheckButtons(o_ax, o_labels, [state[k] for k in o_keys])

    def _on_option(label):
        state[o_keys[o_labels.index(label)]] = not state[o_keys[o_labels.index(label)]]
        _mark_stale()
    o_check.on_clicked(_on_option)

    # --- 計算ボタン (右下。重い処理はここでだけ走る) ----------------------------
    b_ax = fig.add_axes([0.80, 0.08, 0.18, 0.09])
    button = Button(b_ax, L("計算 (Compute)", "Compute"), color="#cfe8cf",
                    hovercolor="#a8d8a8")
    button.on_clicked(_compute)

    note = L("左ラジオでパラメータ群を切替→スライダで上書き。スライダ/チェックは即時計算しません: "
             "設定後に [計算] を押す (水深%g–%gm を MC=%dseeds で掃引)。実効設定は results/explore/config.toml "
             "に出力。SBL/光学なしは深度必須。閉じると終了。"
             % (min(depths), max(depths), n_seeds),
             "Pick a param group (left radio), tweak sliders, then press [Compute]. "
             "Sweeps depth %g-%gm (MC=%dseeds). Effective settings -> results/explore/config.toml."
             % (min(depths), max(depths), n_seeds))
    fig.text(0.30, 0.02, note, fontsize=9)
    # GC 防止に参照保持
    fig._explore_widgets = (slot_sliders, cat_radio, loss_radio, m_check, o_check, button)
    _apply_group(GROUP_ORDER[0])         # 初期ページのスライダを構成
    # 初回: 一度だけ計算して表示し、README・図・config.toml を results/explore/ に保存
    curves0 = compute_curves(state, depths, n_seeds, seed, workers=workers)
    _draw_curves(ax, depths, curves0)
    title.set_text(_cond_text(state))
    _write_report_explore(state, depths, curves0, n_seeds, seed)
    print(f"[explore] README/図/config.toml を出力: {scenario_dir('explore')}")
    plt.show()


def _apply_cli_overrides(state, args):
    """CLI の個別オプション / --set KEY=VALUE を state に反映する (config.toml は不変)。"""
    # 利便のための個別オプション (未指定=None は無視し、config 初期値を残す)
    conv = {"clarity": args.clarity, "sigma_az": args.sigma_az, "sigma_el": args.sigma_el,
            "sigma_dist": args.sigma_dist, "sbl_baseline": args.sbl_baseline}
    for k, v in conv.items():
        if v is not None:
            state[k] = float(v)
    # 汎用 --set KEY=VALUE (全数値パラメータを上書き可能)
    for item in (args.set or []):
        if "=" not in item:
            raise SystemExit(f"[explore] --set は KEY=VALUE 形式です: '{item}'")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in DEFAULTS:
            keys = ", ".join(DEFAULTS.keys())
            raise SystemExit(f"[explore] --set の不明なキー '{k}'。指定可能: {keys}")
        try:
            state[k] = float(v)
        except ValueError:
            raise SystemExit(f"[explore] --set の値が数値ではありません: '{item}'")
    # ON/OFF・損失・手法 (これらは常に値を持つ)
    state["use_depth"] = args.depth
    state["use_error"] = args.error
    state["use_atten"] = args.attenuation
    state["use_attitude"] = args.attitude
    state["att_correct"] = args.att_correct
    state["loss"] = args.loss
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    state["m_optical"] = "optical" in methods
    state["m_sbl"] = "sbl" in methods
    state["m_no_optical"] = "no_optical" in methods
    return state


def main():
    d0 = initial_state()
    ap = argparse.ArgumentParser(
        description="測位手法 (光学/SBL/光学なし) 対話探索。config.toml を書き換えずに"
                    "全誤差モデルを上書きでき、実効設定を results/explore/config.toml に出力する。")
    ap.add_argument("--once", action="store_true", help="GUI を開かず一発計算する")
    ap.add_argument("--list-params", action="store_true",
                    help="調整可能パラメータの一覧 (キー・説明・初期値) を表示して終了する")
    ap.add_argument("--depths", default=",".join(str(d) for d in DEPTH_GRID),
                    help="水深グリッド [m] (カンマ区切り。横軸)")
    # 利便のための個別オプション (既定 None=config 初期値を使う)
    ap.add_argument("--clarity", type=float, default=None, help="濁り c [1/m] (光学)")
    ap.add_argument("--sigma-az", type=float, default=None, help="方位角ノイズ [deg]")
    ap.add_argument("--sigma-el", type=float, default=None, help="仰角ノイズ [deg]")
    ap.add_argument("--sigma-dist", type=float, default=None, help="距離ノイズ [m]")
    ap.add_argument("--sbl-baseline", type=float, default=None, help="SBL アレイ一辺 [m]")
    # 汎用上書き: 全数値パラメータを KEY=VALUE で (繰り返し可)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                    help="任意の誤差パラメータを上書き (繰り返し可)。KEY 一覧: "
                         + ", ".join(DEFAULTS.keys()))
    ap.add_argument("--loss", choices=LOSS_CHOICES, default=d0["loss"],
                    help="ロバスト損失 §4.4 (初期=config [estimator].loss)")
    # ON/OFF は config の enable を既定にし、--xxx / --no-xxx で上書き (BooleanOptionalAction)
    ap.add_argument("--depth", action=argparse.BooleanOptionalAction, default=d0["use_depth"],
                    help="深度センサ§10 を使う (--no-depth で無効。初期=config [depth].enable)")
    ap.add_argument("--error", action=argparse.BooleanOptionalAction, default=d0["use_error"],
                    help="現実誤差§8 を使う (--no-error で無効。初期=config [error_model].enable)")
    ap.add_argument("--attenuation", action=argparse.BooleanOptionalAction,
                    default=d0["use_atten"],
                    help="光減衰§9 を使う (--no-attenuation で無効。初期=config [optical].enable)")
    ap.add_argument("--attitude", action=argparse.BooleanOptionalAction,
                    default=d0["use_attitude"],
                    help="波動揺§14 を光学角度の誤差として適用 (--no-attitude で無効。初期=config [attitude].as_error)")
    ap.add_argument("--att-correct", action=argparse.BooleanOptionalAction,
                    default=d0["att_correct"],
                    help="波動揺適用時に IMU 相補フィルタで姿勢補正 (--no-att-correct で naive。初期=config [attitude].imu_correct)")
    ap.add_argument("--methods", default="optical,sbl,no_optical",
                    help="計算する手法 (カンマ区切り: optical,sbl,no_optical)")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS, help="平均シード数")
    ap.add_argument("--workers", type=int, default=None,
                    help="CPU 並列ワーカー数 (既定=自動=CPUコア数。1 で逐次実行)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.list_params:
        _print_params()
        return

    global WORKERS
    WORKERS = args.workers
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    state = _apply_cli_overrides(initial_state(), args)

    if not CONFIG_TOML_LOADED:
        print("[explore] config.toml 未読込 (Python 3.10 等)。src/config.py のデフォルトを初期値に使用。")
    if args.once or not _INTERACTIVE:
        if not _INTERACTIVE and not args.once:
            print("[explore] GUI バックエンドが無いため一発計算にフォールバックします "
                  "(対話 UI には Tk/Qt が必要)。")
        _one_shot(state, depths, args.n_seeds, args.seed, workers=args.workers)
    else:
        print(f"[explore] 対話 UI を起動 (backend={matplotlib.get_backend()}, "
              f"MC={args.n_seeds}seeds, workers={_resolve_workers(args.workers)})。"
              "ウィンドウを閉じると終了します。")
        _interactive(state, depths, args.n_seeds, args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
