"""attitude.py — 親機姿勢の純粋回転数学 + IMU 姿勢推定 (MATH_SPEC §14)。

親機は水上に浮かび、波で不規則に動揺する。機体固定のカメラが測る方位/仰角は
**機体フレーム**の角度になり、ワールド ENU とズレる。これを IMU (ジャイロ+加速度+磁気)
の姿勢推定で補正する。本モジュールは:

  - 回転の表現変換 (Euler ZYX <-> 回転行列, 回転ベクトルの Exp/Log)
  - 加速度(重力)+磁気からの姿勢算出 attitude_from_accel_mag
  - SO(3) 相補フィルタ complementary_filter (ジャイロ予測 + 加速度/磁気補正)
  - 機体角度 -> ワールド角度の観測補正 body_bearing_to_world

注意 (MBD): 本モジュールは姿勢『推定』側であり truth を一切 import しない
(test_separation が強制)。入力は IMU 信号 (センサ値) と既知パラメータのみ。
回転規約は MATH_SPEC §14.0: R は body->world、Euler は ZYX (yaw->pitch->roll)。
"""
import numpy as np


# ----------------------------------------------------------------------------
# Euler (ZYX) <-> 回転行列  (MATH_SPEC §14.0)
#   R = Rz(yaw) Ry(pitch) Rx(roll),  v_world = R v_body
# ----------------------------------------------------------------------------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_to_matrix(roll, pitch, yaw):
    """Euler ZYX (roll, pitch, yaw) [rad] -> 回転行列 R (body->world)  (MATH_SPEC §14.0)。

    R = Rz(yaw) Ry(pitch) Rx(roll)。roll=pitch=yaw=0 で R=I (機体=ワールド, §0 前提)。
    """
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_euler(R):
    """回転行列 R (body->world) -> Euler ZYX (roll, pitch, yaw) [rad]  (MATH_SPEC §14.0)。

    euler_to_matrix の逆。pitch=+-90deg 近傍 (ジンバルロック) では yaw/roll が縮退するが、
    波の動揺は小角なので問題にならない。
    """
    R = np.asarray(R, dtype=float)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


# ----------------------------------------------------------------------------
# 回転ベクトルの Exp / Log (Rodrigues)  (MATH_SPEC §14.4)
#   姿勢の予測 (ジャイロ積分) と SO(3) 上の補正に使う。
# ----------------------------------------------------------------------------
def _skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def exp_so3(rotvec):
    """回転ベクトル (軸*角度) [rad] -> 回転行列 (Rodrigues の公式)  (MATH_SPEC §14.4)。"""
    rotvec = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3) + _skew(rotvec)        # 1次近似 (theta~0 で安定)
    K = _skew(rotvec / theta)
    return (np.eye(3) + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K))


def log_so3(R):
    """回転行列 -> 回転ベクトル (軸*角度) [rad]  (exp_so3 の逆, MATH_SPEC §14.4)。

    theta = acos((tr(R)-1)/2)。theta~0 では 0、theta~pi 近傍は縮退するが動揺では使わない。
    """
    R = np.asarray(R, dtype=float)
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-12:
        return np.zeros(3)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * np.sin(theta)) * vee


# ----------------------------------------------------------------------------
# 加速度(重力) + 磁気 からの姿勢算出  (MATH_SPEC §14.3)
#   静止時の加速度計は重力基準 acc = R^T (0,0,g) を読む -> roll/pitch を絶対に決める。
#   磁気は mag = R^T m_W を読む -> 傾き補正して yaw を決める。
# ----------------------------------------------------------------------------
def roll_pitch_from_accel(acc):
    """加速度計の読み (重力基準, 機体フレーム) から roll, pitch [rad] を求める (MATH_SPEC §14.3)。

    acc = R^T (0,0,g) のとき roll=atan2(a_y,a_z), pitch=atan2(-a_x, hypot(a_y,a_z))。
    大きさ g は比に効かないのでキャンセルする (方向のみ使用)。
    """
    ax, ay, az = np.asarray(acc, dtype=float)
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.hypot(ay, az))
    return roll, pitch


def yaw_from_mag(mag, roll, pitch):
    """磁気の読み (機体フレーム) を roll/pitch で傾き補正して yaw [rad] を求める (MATH_SPEC §14.3)。

    水平に戻した磁気 m_lvl = Ry(pitch) Rx(roll) mag は (sin yaw, cos yaw, 0) になるので
    yaw = atan2(m_lvl_x, m_lvl_y) (磁気基準 m_W=(0,1,0)=北)。
    """
    m_lvl = rot_y(pitch) @ rot_x(roll) @ np.asarray(mag, dtype=float)
    return np.arctan2(m_lvl[0], m_lvl[1])


def attitude_from_accel_mag(acc, mag):
    """加速度+磁気から Euler (roll, pitch, yaw) [rad] を算出する (MATH_SPEC §14.3)。

    ノイズが無ければ真の姿勢を厳密に復元する (加速度で roll/pitch, 磁気で yaw)。
    相補フィルタの『測定姿勢』として使う。
    """
    roll, pitch = roll_pitch_from_accel(acc)
    yaw = yaw_from_mag(mag, roll, pitch)
    return np.array([roll, pitch, yaw])


# ----------------------------------------------------------------------------
# SO(3) 相補フィルタ  (MATH_SPEC §14.4)
#   ジャイロを積分した予測 R_pred と、加速度/磁気の測定姿勢 R_meas を SO(3) 上で混ぜる:
#     R_pred = R_{k-1} Exp(gyro_{k-1} dt)
#     R_k    = R_pred Exp( (1-alpha) Log(R_pred^T R_meas) )
#   alpha=1 でジャイロのみ (ドリフトする)、alpha 小で加速度/磁気を強く信頼。
# ----------------------------------------------------------------------------
def complementary_filter(gyro_seq, acc_seq, mag_seq, dt, alpha=0.98, R0=None):
    """IMU 信号列から姿勢列 R_seq (n,3,3) を推定する (MATH_SPEC §14.4)。

    gyro_seq: (n-1,3) 各区間 [t_k, t_{k+1}] の機体角速度 [rad/s]
    acc_seq : (n,3)   各時刻の加速度計読み (重力基準) [m/s^2]
    mag_seq : (n,3)   各時刻の磁気読み (機体フレーム)
    dt      : サンプル間隔 [s]
    alpha   : 相補係数 [0,1] (1=ジャイロのみ, 小=加速度/磁気依存)
    R0      : 初期姿勢 (3,3)。None なら acc/mag[0] から算出 (attitude_from_accel_mag)。
    戻り値  : R_seq (n,3,3) body->world の推定姿勢列。truth は参照しない (MBD)。
    """
    acc_seq = np.asarray(acc_seq, dtype=float)
    mag_seq = np.asarray(mag_seq, dtype=float)
    gyro_seq = np.asarray(gyro_seq, dtype=float)
    n = len(acc_seq)
    R_seq = np.empty((n, 3, 3))
    if R0 is None:
        R_seq[0] = euler_to_matrix(*attitude_from_accel_mag(acc_seq[0], mag_seq[0]))
    else:
        R_seq[0] = np.asarray(R0, dtype=float)
    for k in range(1, n):
        R_pred = R_seq[k - 1] @ exp_so3(gyro_seq[k - 1] * dt)     # ジャイロ予測
        R_meas = euler_to_matrix(*attitude_from_accel_mag(acc_seq[k], mag_seq[k]))
        corr = log_so3(R_pred.T @ R_meas)                        # 予測->測定の差 (回転ベクトル)
        R_seq[k] = R_pred @ exp_so3((1.0 - alpha) * corr)        # SO(3) 上で混合
    return R_seq


def euler_sequence(R_seq):
    """姿勢行列列 (n,3,3) -> Euler 列 (n,3) [rad] (roll,pitch,yaw)。可視化・評価用。"""
    R_seq = np.asarray(R_seq, dtype=float)
    return np.array([matrix_to_euler(R) for R in R_seq])


# ----------------------------------------------------------------------------
# 機体角度 -> ワールド角度の観測補正  (MATH_SPEC §14.5)
#   機体で測った (az_B, el_B) を推定姿勢 R_est でワールドへ戻す。距離 d は回転不変。
#   補正後の (d, az_W, el_W) はそのまま既存の estimator (§4, §5) に渡せる。
# ----------------------------------------------------------------------------
def body_bearing_to_world(z_body, R):
    """機体観測 (d, az_B, el_B) を姿勢 R でワールド観測 (d, az_W, el_W) に変換する (MATH_SPEC §14.5)。

    視線単位ベクトル u_B を作り u_W = R u_B に回し、ワールドの方位/仰角に戻す。
    R=I なら入力と一致 (後方互換)。R=R_true ならノイズ無しでワールド観測 (§1) を厳密復元する。
    """
    d, az, el = np.asarray(z_body, dtype=float)
    u_B = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    u_W = np.asarray(R, dtype=float) @ u_B
    az_w = np.arctan2(u_W[1], u_W[0])
    el_w = np.arctan2(u_W[2], np.hypot(u_W[0], u_W[1]))
    return np.array([d, az_w, el_w])


def correct_observation_sequence(z_seq_body, R_seq):
    """機体観測列 (n,3) を姿勢列 (n,3,3) でワールド観測列 (n,3) に補正する (MATH_SPEC §14.5)。"""
    z_seq_body = np.asarray(z_seq_body, dtype=float)
    R_seq = np.asarray(R_seq, dtype=float)
    out = np.empty_like(z_seq_body)
    for k in range(len(z_seq_body)):
        out[k] = body_bearing_to_world(z_seq_body[k], R_seq[k])
    return out
