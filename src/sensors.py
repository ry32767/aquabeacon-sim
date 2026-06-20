"""sensors.py — ② センサモデル (MATH_SPEC §1, §2)。

真値を知ってよい層。真の相対ベクトル/位置から、センサが測るはずの観測量
(距離・方位角・仰角) を計算し、必要ならノイズを加える。

- forward_observation : 位置(相対ベクトル) -> 観測量          (MATH_SPEC §1)
- inverse_observation : 観測量 -> 位置(相対ベクトル)          (MATH_SPEC §2)
- relative_vector     : 親機/子機位置 -> 相対ベクトル v       (MATH_SPEC §0.3 を集約)
- simulate_observation: 真値 + seed -> ノイズ付き観測         (Stage 1)
"""
import numpy as np


def relative_vector(p_child, p_parent=None, observe_from_parent=True):
    """親機・子機の位置から観測の相対ベクトル v=(vx,vy,vz) を作る (MATH_SPEC §0.3)。

    observe_from_parent=True  (親機→子機): v = p_child - p_parent
    observe_from_parent=False (子機→親機): v = p_parent - p_child
    観測の向き定義をここ1か所に集約する。
    """
    p_child = np.asarray(p_child, dtype=float)
    if p_parent is None:
        p_parent = np.zeros(3)
    p_parent = np.asarray(p_parent, dtype=float)
    return (p_child - p_parent) if observe_from_parent else (p_parent - p_child)


def forward_observation(v):
    """相対ベクトル v=(vx,vy,vz) [m] -> 観測 (d [m], theta [rad], phi [rad])  (MATH_SPEC §1)。

    theta: 方位角 azimuth, 範囲 (-pi, pi]。
    phi:   仰角 elevation, 子機が下なら負。
    規約: v=(0,0,0) では theta=phi=0 (atan2(0,0)=0)。
    """
    vx, vy, vz = v
    d = np.sqrt(vx**2 + vy**2 + vz**2)
    theta = np.arctan2(vy, vx)
    phi = np.arctan2(vz, np.hypot(vx, vy))
    return np.array([d, theta, phi])


def inverse_observation(d, theta, phi):
    """観測 (d, theta, phi) -> 相対ベクトル v=(vx,vy,vz) [m]  (MATH_SPEC §2)。

    球面座標→直交座標。ノイズが無ければこれだけで真値に一致する。
    """
    return np.array([
        d * np.cos(phi) * np.cos(theta),
        d * np.cos(phi) * np.sin(theta),
        d * np.sin(phi),
    ])


def simulate_observation(p_child, sigma, seed, p_parent=None,
                         observe_from_parent=True):
    """真の子機位置から、ノイズ付き観測 (d, theta, phi) を生成する (Stage 1)。

    p_child : 真の子機位置 [m] (3,)
    sigma   : (sigma_dist [m], sigma_az [rad], sigma_el [rad])
    seed    : 乱数シード (再現性のため必須)
    戻り値  : ノイズ付き観測 z=(d, theta, phi)

    ノイズは各観測成分に独立な正規分布 N(0, sigma) を加える。
    角度成分にノイズを足した結果は (-pi,pi] の外に出うるが、推定側の残差で
    wrap_angle により吸収されるため、ここでは正規化しない。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    v = relative_vector(p_child, p_parent, observe_from_parent)
    z_true = forward_observation(v)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.asarray(sigma, dtype=float), size=3)
    return z_true + noise


def simulate_observation_sequence(trajectory, sigma, seed, p_parent=None,
                                  observe_from_parent=True):
    """軌道 (n,3) の各時刻にノイズ付き観測を生成して (n,3) で返す (Stage 2)。

    各時刻 k には seed+k を使い、独立かつ再現可能なノイズを与える。
    """
    trajectory = np.asarray(trajectory, dtype=float)
    z = np.empty_like(trajectory)
    for k, p in enumerate(trajectory):
        z[k] = simulate_observation(p, sigma, seed=seed + k, p_parent=p_parent,
                                    observe_from_parent=observe_from_parent)
    return z


# ----------------------------------------------------------------------------
# 現実的センサ誤差モデル (MATH_SPEC §8)
#
# 理想モデル (simulate_observation) に、実機で効く誤差源を重ねる:
#   §8.1 系統バイアス        : z に定数オフセット (取付・校正誤差)
#   §8.2 距離依存ノイズ      : σ(d) = σ0 * (1 + k*d) (遠いほど悪化)
#   §8.3 外れ値              : 確率 p で大きな誤差 (ライト見失い・音響マルチパス)
#   §8.4 音速ズレ            : d_meas = d_true * (c_assumed / c_true) (距離の系統スケール)
#   §8.5 時刻同期            : 音響は latency 秒前の位置の距離 (その間に子機が動く)
#
# すべての既定値は『理想』で、simulate_observation と完全一致する
#   (bias=0, growth=0, outlier_rate=0, c_true=c_assumed, latency=0)。
# ----------------------------------------------------------------------------
def effective_sigma(d, sigma, range_growth_per_m=0.0, dist_growth_per_m=0.0):
    """距離 d における有効ノイズ標準偏差 (σ_d, σ_az, σ_el) を返す (MATH_SPEC §8.2)。

    σ_d(d)   = σ_d0   * (1 + dist_growth_per_m  * d)
    σ_ang(d) = σ_ang0 * (1 + range_growth_per_m * d)   (方位・仰角の両方)
    growth=0 なら sigma をそのまま返す (理想)。
    """
    sd, saz, sel = sigma
    fa = 1.0 + range_growth_per_m * d
    fd = 1.0 + dist_growth_per_m * d
    return np.array([sd * fd, saz * fa, sel * fa])


def simulate_observation_realistic(p_child, sigma, seed, p_parent=None,
                                   observe_from_parent=True, *,
                                   bias=(0.0, 0.0, 0.0),
                                   range_growth_per_m=0.0,
                                   dist_growth_per_m=0.0,
                                   outlier_rate=0.0,
                                   outlier_scale=20.0,
                                   sound_speed_true=1500.0,
                                   sound_speed_assumed=1500.0,
                                   acoustic_latency_s=0.0,
                                   velocity=None):
    """現実的な誤差を含むノイズ付き観測 (d, theta, phi) を生成する (MATH_SPEC §8)。

    既定値はすべて『理想』で、simulate_observation(同 seed) と一致する。
    config.ERROR_MODEL をキーワード展開してそのまま渡せる:
        simulate_observation_realistic(p, SIGMA, seed, **ERROR_MODEL)

    p_child            : 真の子機位置 [m] (光学観測の時刻 t における位置)
    velocity           : 子機速度 [m/s] (3,)。時刻同期 (§8.5) で使用。None で 0。
    acoustic_latency_s : 音響が光学より遅れる時間 [s]。音響距離は t-latency の位置で測る。
    その他のキーワードは §8 各項を参照。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    if velocity is None:
        velocity = np.zeros(3)
    velocity = np.asarray(velocity, dtype=float)
    rng = np.random.default_rng(seed)

    # --- 角度は光学時刻 t の位置から、距離は音響時刻 t-latency の位置から (§8.5) ---
    v_opt = relative_vector(p_child, p_parent, observe_from_parent)
    _, theta_true, phi_true = forward_observation(v_opt)
    d_optical = np.linalg.norm(v_opt)                 # 有効σの距離基準にはこちらを使う

    p_acoustic = np.asarray(p_child, float) - velocity * acoustic_latency_s
    v_aco = relative_vector(p_acoustic, p_parent, observe_from_parent)
    d_true = np.linalg.norm(v_aco)

    # --- 音速ズレ: 測距は飛行時間×仮定音速 = d_true * c_assumed/c_true (§8.4) ---
    d_meas = d_true * (sound_speed_assumed / sound_speed_true)

    z = np.array([d_meas, theta_true, phi_true]) + np.asarray(bias, dtype=float)

    # --- 距離依存ノイズ (§8.2) を有効σとして加える (§7 の零平均ガウス) ---
    eff = effective_sigma(d_optical, sigma, range_growth_per_m, dist_growth_per_m)
    z = z + rng.normal(0.0, eff, size=3)

    # --- 外れ値 (§8.3): 各成分が確率 outlier_rate で大きく飛ぶ ---
    if outlier_rate > 0.0:
        for i in range(3):
            if rng.random() < outlier_rate:
                z[i] += rng.normal(0.0, outlier_scale * eff[i])
    return z


def simulate_observation_sequence_realistic(trajectory, sigma, seed,
                                            p_parent=None,
                                            observe_from_parent=True,
                                            dt=None, **model):
    """軌道 (n,3) に現実的誤差付き観測列を生成して (n,3) で返す (MATH_SPEC §8)。

    時刻同期 (§8.5) のために各時刻の速度を軌道の差分から推定する:
        velocity_k ~= (p_{k+1} - p_k) / dt,  dt は光学サンプリング間隔 [s]。
    dt=None なら 1/OPTICAL_RATE_HZ を使う。model は simulate_observation_realistic
    のキーワード (config.ERROR_MODEL を ** 展開して渡せる)。
    """
    trajectory = np.asarray(trajectory, dtype=float)
    n = len(trajectory)
    if dt is None:
        from src.config import OPTICAL_RATE_HZ
        dt = 1.0 / OPTICAL_RATE_HZ
    # 前進差分で速度を近似 (末端は後退差分)
    vel = np.gradient(trajectory, dt, axis=0) if n >= 2 else np.zeros_like(trajectory)
    z = np.empty_like(trajectory)
    for k, p in enumerate(trajectory):
        z[k] = simulate_observation_realistic(
            p, sigma, seed=seed + k, p_parent=p_parent,
            observe_from_parent=observe_from_parent, velocity=vel[k], **model)
    return z


def _bearing(v):
    """相対ベクトル v -> (az, el) [rad]。forward_observation の角度部分 (距離は捨てる)。"""
    vx, vy, vz = v
    return np.array([np.arctan2(vy, vx), np.arctan2(vz, np.hypot(vx, vy))])


def stereo_camera_positions(point, center, standoff, baseline, up=None):
    """子機ステレオの左右カメラ位置 (c_L, c_R) を返す (MATH_SPEC §6.2)。

    対象表面点 point を、中心 center から見た外向き法線方向に standoff だけ離れた所から
    正対観測する理想化。ベースラインは視線と直交する向きに baseline だけ取る。

    point   : 観測する表面点 [m]
    center  : 対象のおおよその中心 [m] (視線方向を決めるだけ。接近時に既知)
    standoff: 表面からの観測距離 [m]
    baseline: 左右カメラ間隔 [m]
    up      : ベースライン方向を決める補助軸 (視線と直交化)。None なら [0,0,1]。
    戻り値  : (c_L, c_R) 各 (3,)

    注意: これは「各点を法線方向から standoff で正対」する簡易モデル。実機の子機は
    標準オフ軌道を飛んで撮るが、ステレオ精度 (距離・ベースライン依存) を見るには十分。
    オクルージョンは考慮しない。
    """
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    up = np.asarray(up, dtype=float)

    view = point - center
    nrm = np.linalg.norm(view)
    view_dir = view / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])

    # ベースラインは視線と直交させる (ステレオの基本)
    b_dir = np.cross(view_dir, up)
    if np.linalg.norm(b_dir) < 1e-9:        # 視線と up が平行なら別軸で直交化
        b_dir = np.cross(view_dir, np.array([1.0, 0.0, 0.0]))
    b_dir = b_dir / np.linalg.norm(b_dir)

    rig = point + standoff * view_dir       # カメラリグの中心 (表面から standoff)
    c_L = rig - 0.5 * baseline * b_dir
    c_R = rig + 0.5 * baseline * b_dir
    return c_L, c_R


def simulate_stereo_observation(point, cam_L, cam_R, sigma_cam, seed):
    """ステレオ2カメラのノイズ付き観測 (az_L, el_L, az_R, el_R) を返す (MATH_SPEC §6.2)。

    真の表面点 point を左右カメラ (cam_L, cam_R) が角度で観測する。各カメラの
    (az, el) に独立な正規ノイズ N(0, sigma_cam) を加える。距離は測らない (角度のみ)。
    seed で再現可能。
    """
    point = np.asarray(point, dtype=float)
    rng = np.random.default_rng(seed)
    bL = _bearing(point - np.asarray(cam_L, float)) + rng.normal(0.0, sigma_cam, 2)
    bR = _bearing(point - np.asarray(cam_R, float)) + rng.normal(0.0, sigma_cam, 2)
    return np.array([bL[0], bL[1], bR[0], bR[1]])


def simulate_imu_displacements(trajectory, sigma_imu, seed):
    """IMU pre-integration による時刻間変位 delta_p の擬似観測を返す (n-1, 3) [m]  (MATH_SPEC §5)。

    真の変位 (p_{k+1} - p_k) に正規ノイズ N(0, sigma_imu) を加えたもの。
    IMU 加速度の二重積分による予測変位の代理モデル (簡易版)。seed で再現可能。

    trajectory: (n,3) 真の子機軌道
    sigma_imu : スカラ or (3,) の変位ノイズ標準偏差 [m]
    戻り値    : delta_meas (n-1, 3)
    """
    trajectory = np.asarray(trajectory, dtype=float)
    true_delta = np.diff(trajectory, axis=0)        # (n-1, 3)
    sigma_imu = np.broadcast_to(np.asarray(sigma_imu, dtype=float), (3,))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma_imu, size=true_delta.shape)
    return true_delta + noise
