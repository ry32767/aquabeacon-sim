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
                                  observe_from_parent=True, rho=0.0):
    """軌道 (n,3) の各時刻にノイズ付き観測を生成して (n,3) で返す (Stage 2)。

    各時刻 k には seed+k を使い、独立かつ再現可能なノイズを与える。
    rho>0 (スカラ or (3,)) なら時間相関ノイズ (1次ガウス・マルコフ, §8.6) を与える。
    既定 rho=0 では従来の白色・per-step seed と完全一致 (後方互換)。
    """
    trajectory = np.asarray(trajectory, dtype=float)
    rho_arr = np.broadcast_to(np.asarray(rho, dtype=float), (3,))
    if np.any(rho_arr != 0.0):                   # 時間相関ノイズ (§8.6): 単一連続ストリーム
        z = np.empty_like(trajectory)
        e = gauss_markov_sequence(len(trajectory), np.asarray(sigma, dtype=float),
                                  rho_arr, seed=seed, per_step_shape=(3,))
        for k, p in enumerate(trajectory):
            v = relative_vector(p, p_parent, observe_from_parent)
            z[k] = forward_observation(v) + e[k]
        return z
    z = np.empty_like(trajectory)                # 白色 (従来パス, 後方互換)
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


def effective_sound_speed(c0, gradient_per_s, z_child, z_parent=0.0):
    """直線スラント経路の飛行時間に効く実効音速 [m/s] (SVP1次近似, MATH_SPEC §8.4b)。

    深さ方向に線形な音速プロファイル c(z)=c0+gradient·z (z は上向き, 子機は z<0) のとき、
    親機 (z_parent=0) から子機 (z_child) への直線経路に沿った飛行時間 TOF=∫ds/c は
    対数平均音速 c_eff=(c_child-c0)/ln(c_child/c0) で表せる。測距は d_meas=d_true·c_assumed/c_eff
    になる (既存 §8.4 の定数スケールを経路依存に一般化した追加項)。gradient=0 で c_eff=c0
    (従来と完全一致)。near-nadir では経路がほぼ鉛直で効果は小さい (ray bending は2次で無視)。
    """
    if gradient_per_s == 0.0 or z_child == z_parent:
        return c0
    c_child = c0 + gradient_per_s * (z_child - z_parent)
    if c_child <= 0.0 or c0 <= 0.0:              # 非物理ガード
        return c0
    return (c_child - c0) / np.log(c_child / c0)


def gauss_markov_sequence(n, sigma, rho, seed=None, per_step_shape=(), rng=None):
    """定常 1次ガウス・マルコフ (AR(1)) ノイズ列を返す (MATH_SPEC §8.6)。

        e_0 = sigma · w_0,   e_k = rho · e_{k-1} + sqrt(1-rho²) · sigma · w_k,   w ~ N(0,1)

    定常なので周辺分散は sigma² (全時刻一定)、lag-1 自己相関は rho。rho=0 で白色 N(0,sigma)。
    実機の光学重心追跡・音響測距の誤差は時間相関を持つ (rho>0) ので、白色のみだと平滑化/IMU
    融合の利得を過大評価する。これはその相関を理想 (白色) に重ねるための基本生成器。

    n            : 時刻数
    sigma        : 標準偏差 (スカラ or per_step_shape にブロードキャスト可)
    rho          : lag-1 自己相関 [0,1) (スカラ or per_step_shape)
    per_step_shape: 1時刻あたりの成分形状。観測 (3,) / 深度 () / SBL (M,)。
    rng          : 与えれば単一連続ストリームを使う (独立サブストリーム化, §15.2)。
                   None なら default_rng(seed)。
    戻り値       : (n,)+per_step_shape の相関ノイズ列。
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    shape = (n,) + tuple(per_step_shape)
    w = rng.standard_normal(shape)
    e = np.empty(shape)
    e[0] = w[0]                                  # 定常開始 (分散1)
    s = np.sqrt(1.0 - rho**2)
    for k in range(1, n):
        e[k] = rho * e[k - 1] + s * w[k]
    return e * sigma


# ----------------------------------------------------------------------------
# 親機の光学リンク: 水中の減衰・拡散モデル (MATH_SPEC §9)
#
# 光は水中で減衰 (吸収 a + 散乱 b、c = a+b) し、距離とともに受光信号が落ちる:
#   透過率 T(d)   = exp(-c·d)
#   信号比 R(d)   = (d_ref/d)^2 · exp(-c·(d - d_ref))   (幾何拡散 1/d^2 × 差分透過)
#   SNR(d)        = snr_ref · R(d)^p                    (p=1 後方散乱律速 / 0.5 ショット律速)
#   角度ノイズ σ_ang(d) = σ_floor + (σ_ref - σ_floor) / R(d)^p
#       (SNR が下がるほど重心推定が甘くなり角度精度が悪化。d_ref で σ_ref に一致)
#   ドロップアウト確率 p_drop(d): SNR < snr_min で見失い (誤検出=角度の外れ値)
#
# model は config.OPTICAL_MODEL と同じキーの辞書 (角度は rad, 距離は m, c は 1/m)。
# ----------------------------------------------------------------------------
def optical_signal_ratio(d, attenuation_c, range_ref):
    """受光信号比 R(d) = (d_ref/d)^2 · exp(-c·(d-d_ref))。d=range_ref で 1。"""
    d = max(float(d), 1e-9)
    return (range_ref / d) ** 2 * np.exp(-attenuation_c * (d - range_ref))


def optical_snr(d, model):
    """距離 d における光学 SNR (MATH_SPEC §9)。"""
    R = optical_signal_ratio(d, model["attenuation_c"], model["range_ref"])
    return model["snr_ref"] * R ** model["snr_exponent"]


def optical_angular_sigma(d, model):
    """距離 d における有効角度ノイズ σ_ang(d) [rad] (MATH_SPEC §9)。

    σ_floor + (σ_ref - σ_floor)/R^p。d=range_ref で σ_ref、近いほど σ_floor に漸近、
    遠い/濁るほど増大する。
    """
    R = optical_signal_ratio(d, model["attenuation_c"], model["range_ref"])
    return model["sigma_floor"] + (model["sigma_ref"] - model["sigma_floor"]) \
        / R ** model["snr_exponent"]


def optical_dropout_prob(d, model):
    """距離 d でビーコンを見失う確率 (MATH_SPEC §9)。

    SNR が snr_min を下回るほど 0→dropout_max へ立ち上がるロジスティック。
    SNR>>snr_min でほぼ 0、SNR=snr_min で dropout_max/2。
    """
    snr = optical_snr(d, model)
    k = 4.0 / max(model["snr_min"], 1e-6)        # 立ち上がりの鋭さ
    x = np.clip(k * (snr - model["snr_min"]), -500.0, 500.0)   # exp のオーバーフロー回避
    return float(model["dropout_max"] / (1.0 + np.exp(x)))


def simulate_optical_detection(trajectory, optical_model, seed, p_parent=None,
                               observe_from_parent=True):
    """各時刻でビーコンを検出できたかの bool 列 (n,) を返す (MATH_SPEC §9, §12)。

    検出失敗 (見失い) は光学リンクのドロップアウト確率 p_drop(d) (§9) に従う。自動切替
    (§12) の判定入力に使う。深い/濁った水ほど未検出が増える。seed で再現可能。

    trajectory   : (n,3) 真の子機軌道
    optical_model: 光学リンク減衰モデル (config.OPTICAL_MODEL と同形)
    戻り値       : (n,) bool。True=検出, False=見失い。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    trajectory = np.asarray(trajectory, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(len(trajectory), dtype=bool)
    for k, p in enumerate(trajectory):
        d = np.linalg.norm(relative_vector(p, p_parent, observe_from_parent))
        out[k] = rng.random() >= optical_dropout_prob(d, optical_model)
    return out


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
                                   svp_gradient_per_s=0.0,
                                   velocity=None,
                                   optical_model=None):
    """現実的な誤差を含むノイズ付き観測 (d, theta, phi) を生成する (MATH_SPEC §8)。

    既定値はすべて『理想』で、simulate_observation(同 seed) と一致する。
    config.ERROR_MODEL をキーワード展開してそのまま渡せる:
        simulate_observation_realistic(p, SIGMA, seed, **ERROR_MODEL)

    p_child            : 真の子機位置 [m] (光学観測の時刻 t における位置)
    velocity           : 子機速度 [m/s] (3,)。時刻同期 (§8.5) で使用。None で 0。
    acoustic_latency_s : 音響が光学より遅れる時間 [s]。音響距離は t-latency の位置で測る。
    optical_model      : 光学リンク減衰モデル (MATH_SPEC §9) の辞書。None で無効。
                         与えると角度ノイズ σ_az/σ_el を距離・濁り依存に置換し、確率的に
                         ビーコン見失い (角度の外れ値) を起こす。config.OPTICAL_MODEL を渡せる。
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

    # --- 音速ズレ: 測距は飛行時間×仮定音速 = d_true * c_assumed/c_eff (§8.4, §8.4b) ---
    # SVP (深さ線形プロファイル) があれば実効音速で c_true を置換 (gradient=0 で従来一致)
    c_eff = effective_sound_speed(sound_speed_true, svp_gradient_per_s,
                                  float(p_acoustic[2]), float(np.asarray(p_parent)[2]))
    d_meas = d_true * (sound_speed_assumed / c_eff)

    z = np.array([d_meas, theta_true, phi_true]) + np.asarray(bias, dtype=float)

    # --- 距離依存ノイズ (§8.2) を有効σとして加える (§7 の零平均ガウス) ---
    eff = effective_sigma(d_optical, sigma, range_growth_per_m, dist_growth_per_m)

    # --- 光学リンク減衰 (§9): 角度σを距離・濁り依存に置換 ---
    if optical_model is not None:
        s_ang = optical_angular_sigma(d_optical, optical_model)
        eff[1] = s_ang
        eff[2] = s_ang

    z = z + rng.normal(0.0, eff, size=3)

    # --- 外れ値 (§8.3): 各成分が確率 outlier_rate で大きく飛ぶ ---
    if outlier_rate > 0.0:
        for i in range(3):
            if rng.random() < outlier_rate:
                z[i] += rng.normal(0.0, outlier_scale * eff[i])

    # --- 光学ドロップアウト (§9): SNR 低下でビーコン見失い -> 角度が飛ぶ外れ値 ---
    if optical_model is not None:
        if rng.random() < optical_dropout_prob(d_optical, optical_model):
            jump = optical_model["dropout_jump"]
            z[1] += rng.uniform(-jump, jump)
            z[2] += rng.uniform(-jump, jump)
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


def simulate_depth(p_child, sigma_depth, seed, bias=0.0):
    """深度センサ(圧力)の観測 depth = -z_child + bias + N(0, sigma_depth) を返す (MATH_SPEC §10)。

    水面 (親機, z=0) を基準とした絶対深度 [m, 下が正]。座標は Z=上なので depth = -z。
    距離・濁りに依存せず鉛直を直接測れるのが特徴 (光学が苦手な深い/濁った水で有効)。

    p_child     : 真の子機位置 [m] (3,)
    sigma_depth : 深度ノイズ標準偏差 [m] (圧力センサ精度)
    bias        : 深度の系統バイアス [m] (海面気圧・潮位ドリフト等)。既定0。
    seed        : 乱数シード (再現性)
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(p_child, dtype=float)[2]
    return -z + bias + rng.normal(0.0, sigma_depth)


def simulate_depth_sequence(trajectory, sigma_depth, seed, bias=0.0, rho=0.0):
    """軌道 (n,3) の各時刻に深度観測を生成して (n,) で返す (MATH_SPEC §10)。

    各時刻 k に seed+k を使い、独立かつ再現可能なノイズを与える。
    rho>0 なら時間相関ノイズ (§8.6)。既定 rho=0 で従来の白色と完全一致 (後方互換)。
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if rho != 0.0:                               # 時間相関 (§8.6): 単一連続ストリーム
        z = np.asarray(trajectory, dtype=float)[:, 2]
        e = gauss_markov_sequence(len(trajectory), sigma_depth, rho, seed=seed)
        return -z + bias + e
    out = np.empty(len(trajectory))              # 白色 (従来パス, 後方互換)
    for k, p in enumerate(trajectory):
        out[k] = simulate_depth(p, sigma_depth, seed=seed + k, bias=bias)
    return out


def simulate_sbl_ranges(p_child, anchors, sigma_range, seed, *,
                        sound_speed_true=1500.0, sound_speed_assumed=1500.0,
                        bias_dist=0.0, dist_growth_per_m=0.0,
                        outlier_rate=0.0, outlier_scale=20.0,
                        svp_gradient_per_s=0.0):
    """SBL: 親機の複数トランスデューサ (既知配置 anchors) への距離観測を返す (MATH_SPEC §13)。

    各トランスデューサ i が子機までの距離 d_i = ||p_child - anchor_i|| を測る (音響飛行時間)。
    4点以上の既知配置への距離 → 多辺測量で光学なしに3D位置が定まる。

    SBL も音響測距なので、§8 の音響誤差を**理想に重ねる追加項**として反映できる
    (既定はすべて理想で、従来の純ガウスと完全一致 = 後方互換):
      §8.4 音速ズレ : d_meas = d_true * (c_assumed / c_true)   (距離の系統スケール)
      §8.1 バイアス : + bias_dist                              (取付・校正残差)
      §8.2 距離依存 : σ_eff = σ_range * (1 + dist_growth_per_m * d_true)
      §8.3 外れ値   : 各アンカー独立に確率 outlier_rate で大誤差 (音響マルチパス)

    p_child     : 真の子機位置 [m] (3,)
    anchors     : (M,3) トランスデューサの既知位置 [m]
    sigma_range : 各測距のノイズ標準偏差 [m]
    seed        : 乱数シード (再現性)
    戻り値      : (M,) 各アンカーへのノイズ付き距離
    """
    p_child = np.asarray(p_child, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    rng = np.random.default_rng(seed)
    d_true = np.linalg.norm(anchors - p_child, axis=1)
    # §8.4 音速ズレ (距離スケール) + §8.4b SVP実効音速 + §8.1 系統バイアス
    c_eff = effective_sound_speed(sound_speed_true, svp_gradient_per_s,
                                  float(p_child[2]), float(anchors[..., 2].mean()))
    d_meas = d_true * (sound_speed_assumed / c_eff) + bias_dist
    # §8.2 距離依存ノイズ (遠いほど悪化)
    sigma_eff = sigma_range * (1.0 + dist_growth_per_m * d_true)
    ranges = d_meas + rng.normal(0.0, sigma_eff)
    # §8.3 外れ値 (音響マルチパス): 各アンカーが独立に跳ねる
    if outlier_rate > 0.0:
        for i in range(len(ranges)):
            if rng.random() < outlier_rate:
                ranges[i] += rng.normal(0.0, outlier_scale * sigma_eff[i])
    return ranges


def simulate_sbl_range_sequence(trajectory, anchors, sigma_range, seed, rho=0.0,
                                **error):
    """軌道 (n,3) の各時刻に SBL 距離観測を生成して (n,M) で返す (MATH_SPEC §13)。

    各時刻 k に seed+k を使い、独立かつ再現可能なノイズを与える。error は
    simulate_sbl_ranges の音響誤差キーワード (§8: sound_speed_*, bias_dist,
    dist_growth_per_m, outlier_rate, outlier_scale, svp_gradient_per_s)。既定は理想で従来と一致。
    rho>0 なら測距ノイズに時間相関 (§8.6) を与える (各アンカー独立に時間相関)。
    既定 rho=0 で従来の白色と完全一致 (後方互換)。

    anchors は (M,3) 固定配置のほか、(n,M,3) の**時刻ごとに動くアンカー列**も受ける
    (親機の波動揺でアレイが回る §13.5/§14 など)。後者は各時刻 k に anchors[k] を使う。
    """
    trajectory = np.asarray(trajectory, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    per_step = anchors.ndim == 3              # (n,M,3): 時刻ごとに動くアンカー (波動揺など)
    n, m = len(trajectory), anchors.shape[-2]
    if rho != 0.0:                            # 時間相関 (§8.6): クリーン距離 + 単一連続ストリーム
        clean_error = {**error, "outlier_rate": 0.0}    # 外れ値は本質的に白色なので相関路から除く
        out = np.empty((n, m))
        for k, p in enumerate(trajectory):
            a_k = anchors[k] if per_step else anchors
            out[k] = simulate_sbl_ranges(p, a_k, 0.0, seed=seed + k, **clean_error)
        e = gauss_markov_sequence(n, sigma_range, rho, seed=seed + 99991,
                                  per_step_shape=(m,))
        return out + e
    out = np.empty((n, m))                    # 白色 (従来パス, 後方互換)
    for k, p in enumerate(trajectory):
        a_k = anchors[k] if per_step else anchors
        out[k] = simulate_sbl_ranges(p, a_k, sigma_range, seed=seed + k, **error)
    return out


# ----------------------------------------------------------------------------
# 親機姿勢と IMU 信号 (MATH_SPEC §14)
#
# 親機が波で動揺すると、機体固定カメラの角度は機体フレーム z_body=forward(R^T v) になる
# (距離 d は回転不変)。IMU はジャイロ(角速度)+加速度(重力基準)+磁気(方位基準)を生信号で
# 出力し、推定側 (attitude.py の相補フィルタ) が姿勢 R を復元する。
#
# 本層は truth (姿勢の真値 R) を知ってよい。R から機体観測と IMU 生信号を作る。
# ----------------------------------------------------------------------------
def simulate_observation_attitude(p_child, R, sigma, seed, p_parent=None,
                                  observe_from_parent=True):
    """動揺する親機 (姿勢 R) の機体フレーム観測 (d, az_B, el_B) を生成する (MATH_SPEC §14.2)。

    機体観測は z_body = forward_observation(R^T v)。距離 d の**真値**は回転不変 (音響は姿勢に
    不感) だが、測距そのものには独立な距離ノイズ N(0, sigma_dist) が乗る。角度 (az_B, el_B)
    には角度ノイズ N(0, sigma_az/el) を加える。R=I なら §1 simulate_observation と完全一致する。

    p_child: 真の子機位置 [m] (3,)
    R      : 親機姿勢 (3,3) body->world (attitude.euler_to_matrix で作る)
    sigma  : (sigma_dist [m], sigma_az [rad], sigma_el [rad])
    戻り値 : 機体観測 z_body=(d, az_B, el_B)
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    R = np.asarray(R, dtype=float)
    v_world = relative_vector(p_child, p_parent, observe_from_parent)
    v_body = R.T @ v_world                      # ワールド相対ベクトル -> 機体フレーム (角度のみ回転)
    z_true = forward_observation(v_body)        # 距離真値は ||v_body||=||v_world|| (回転不変)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.asarray(sigma, dtype=float), size=3)
    return z_true + noise


def simulate_observation_sequence_attitude(trajectory, R_seq, sigma, seed,
                                           p_parent=None, observe_from_parent=True):
    """軌道 (n,3) と親機姿勢列 (n,3,3) から機体フレーム観測列 (n,3) を生成する (MATH_SPEC §14.2)。

    各時刻 k に seed+k を使い、独立かつ再現可能な角度ノイズを与える。R_seq[k] はその時刻の
    親機姿勢 (波で動揺)。R_seq が全て I なら従来 (姿勢補正なし) の観測列と一致する。
    """
    trajectory = np.asarray(trajectory, dtype=float)
    R_seq = np.asarray(R_seq, dtype=float)
    z = np.empty_like(trajectory)
    for k, p in enumerate(trajectory):
        z[k] = simulate_observation_attitude(p, R_seq[k], sigma, seed=seed + k,
                                             p_parent=p_parent,
                                             observe_from_parent=observe_from_parent)
    return z


def simulate_imu_signals(R_seq, dt, seed, gyro_sigma=0.0, gyro_bias=0.0,
                         acc_sigma=0.0, mag_sigma=0.0, gravity=9.80665,
                         mag_ref=(0.0, 1.0, 0.0)):
    """親機姿勢列 (n,3,3) から IMU 生信号 (ジャイロ/加速度/磁気) を生成する (MATH_SPEC §14.3)。

    ジャイロ: 区間 [t_k,t_{k+1}] の機体角速度 omega_k = Log(R_k^T R_{k+1})/dt に
             バイアス + 白色ノイズを加える。(n-1,3)。
    加速度  : 重力基準 acc_k = R_k^T (0,0,g) にノイズ。静止浮体の比力近似 (並進加速度は無視)。
    磁気    : mag_k = R_k^T m_W にノイズ。m_W=mag_ref は世界磁気基準 (既定 (0,1,0)=北)。
    すべて機体フレーム。seed で再現可能。ノイズ/バイアス 0 で理想 (姿勢を厳密復元できる)。

    R_seq     : (n,3,3) 親機姿勢 (truth)
    dt        : サンプル間隔 [s]
    gyro_sigma: ジャイロ白色ノイズ [rad/s]
    gyro_bias : ジャイロ定常バイアス [rad/s] (スカラ or (3,))
    acc_sigma : 加速度ノイズ [m/s^2]
    mag_sigma : 磁気ノイズ [-] (mag_ref と同じスケール)
    gravity   : 重力加速度 [m/s^2]
    mag_ref   : 世界磁気基準ベクトル (3,)
    戻り値    : dict(gyro=(n-1,3), acc=(n,3), mag=(n,3))
    """
    from src.attitude import log_so3
    R_seq = np.asarray(R_seq, dtype=float)
    n = len(R_seq)
    rng = np.random.default_rng(seed)
    g_ref = np.array([0.0, 0.0, float(gravity)])     # 静止時に加速度計が読む重力基準 (上向き +g)
    m_ref = np.asarray(mag_ref, dtype=float)
    gyro_bias = np.broadcast_to(np.asarray(gyro_bias, dtype=float), (3,))

    acc = np.empty((n, 3))
    mag = np.empty((n, 3))
    for k in range(n):
        acc[k] = R_seq[k].T @ g_ref + rng.normal(0.0, acc_sigma, 3)
        mag[k] = R_seq[k].T @ m_ref + rng.normal(0.0, mag_sigma, 3)

    gyro = np.empty((n - 1, 3))
    for k in range(n - 1):
        omega = log_so3(R_seq[k].T @ R_seq[k + 1]) / dt    # 区間の真の機体角速度
        gyro[k] = omega + gyro_bias + rng.normal(0.0, gyro_sigma, 3)
    return {"gyro": gyro, "acc": acc, "mag": mag}


def simulate_imu_displacements(trajectory, sigma_imu, seed, sigma_bias=0.0,
                               bias0=0.0):
    """IMU pre-integration による時刻間変位 delta_p の擬似観測を返す (n-1, 3) [m]  (MATH_SPEC §5, §5.5)。

    真の変位 (p_{k+1} - p_k) に白色ノイズ N(0, sigma_imu) を加える。実機の strapdown
    pre-integration はさらに**緩やかに変動するバイアス** (加速度計バイアス・スケール誤差) に
    支配されるので、それを**追加項**として重ねられる (§5.5):

        delta_meas_k = (p_{k+1}-p_k) + e_k + b_k
        e_k = N(0, sigma_imu)                        (白色, 従来項)
        b_k = b_{k-1} + N(0, sigma_bias),  b_0 = bias0  (バイアスのランダムウォーク)

    sigma_bias=0 かつ bias0=0 (既定) では b_k=0 で従来と**完全一致** (白色ノイズ e_k の
    乱数引きを先に行い byte 一致を保つ)。バイアスありで光学なし/SBL フォールバックの
    精度が現実的に劣化する (白色のみは IMU 拘束を過大評価する)。

    trajectory: (n,3) 真の子機軌道
    sigma_imu : スカラ or (3,) の変位白色ノイズ標準偏差 [m]
    sigma_bias: バイアスのランダムウォーク 1ステップ標準偏差 [m] (スカラ or (3,))。既定0。
    bias0     : 初期バイアス [m] (スカラ or (3,))。既定0。
    戻り値    : delta_meas (n-1, 3)
    """
    trajectory = np.asarray(trajectory, dtype=float)
    true_delta = np.diff(trajectory, axis=0)        # (n-1, 3)
    sigma_imu = np.broadcast_to(np.asarray(sigma_imu, dtype=float), (3,))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma_imu, size=true_delta.shape)   # 白色 e_k (従来と同一の引き)
    out = true_delta + noise
    sigma_bias_v = np.broadcast_to(np.asarray(sigma_bias, dtype=float), (3,))
    if np.any(sigma_bias_v != 0.0) or np.any(np.asarray(bias0, dtype=float) != 0.0):
        steps = rng.normal(0.0, sigma_bias_v, size=true_delta.shape)  # 追加乱数 (既定では未消費)
        steps[0] = 0.0                          # b_0 = bias0 (初回はウォークなし, §5.5)
        walk = np.cumsum(steps, axis=0) + np.asarray(bias0, dtype=float)
        out = out + walk
    return out


# ----------------------------------------------------------------------------
# 波による親機動揺を「観測誤差」として既存のワールド角度観測に重ねる (MATH_SPEC §8/§14)
#
# 親機は水上で波により動揺する。機体固定カメラの角度は機体フレームの値になるので、姿勢を
# 無視すると位置に系統誤差 (特に yaw->方位角) が乗る。run_attitude (§14) はこれを生成時に
# 機体フレームで作るが、ここでは**既に作ったワールド角度観測 z を後処理で機体フレームへ回し**、
# 任意で IMU 相補フィルタの推定姿勢で補正して返す合成可能なヘルパを用意する。これにより各
# シナリオは観測生成の直後に 1 行差し込むだけで「波動揺の誤差」を反映できる (距離は回転不変)。
#
# enable=False (既定) では z をそのまま返す = 従来結果と完全一致 (後方互換)。
# 注: 後処理なので「回転→ノイズ」ではなく「ノイズ→回転」になる近似だが、誤差モデルとして
#     動揺の影響を見るには十分。生成時厳密版が要る §14 専用は run_attitude を使う。
# ----------------------------------------------------------------------------
def apply_attitude_error(z_world_seq, seed, *, enable, imu_correct=True, wave=None,
                         dt=None, gyro_sigma=0.0, gyro_bias=0.0, acc_sigma=0.0,
                         mag_sigma=0.0, gravity=9.80665, filter_alpha=0.98):
    """ワールド角度観測列 (n,3) に波動揺の誤差を重ねて返す (MATH_SPEC §8/§14)。

    z_world_seq : (n,3) 既存のワールド観測 (d, az_W, el_W)。距離 d は姿勢に不感。
    enable      : False なら何もしない (z をそのまま返す = 従来と一致)。
    imu_correct : True なら親機 IMU (ジャイロ+加速度+磁気) の相補フィルタ姿勢で機体角度を
                  ワールドへ補正 (姿勢推定の残差ぶんだけ誤差が残る)。False なら naive
                  (機体角度をワールドと誤認 -> 姿勢ぶんの系統誤差が丸ごと残る)。
    wave        : wave_attitude_sequence のキーワード (roll_amp 等 [rad]/[s])。None で config。
    dt          : サンプル間隔 [s] (None で config.ATT_DT)。軌道点を時刻列とみなす。
    gyro_*/acc_*/mag_*/gravity/filter_alpha : IMU 生信号と相補フィルタの設定 (imu_correct 時)。
    戻り値      : (n,3) 補正後 (または naive) のワールド観測。
    truth (親機姿勢 R_true) は本ヘルパ内で生成するが、補正に使う姿勢は IMU からの R_est であり
    推定 (estimator) には truth を渡さない (MBD 分離は保たれる)。

    **注意 (角度誤差統計の妥当性)**: 本ヘルパは「ワールド角度にノイズ→機体へ回転」の後処理
    近似 (NUM-01)。near-nadir (仰角→-90°) では機体方位ノイズ共分散が回転不変でないため、
    この経路の**方位/仰角の誤差統計は不正確**になりうる (位置 RMSE への影響は小)。波動揺の
    角度レベルの誤差・naive vs 補正の角度比較は、生成時に機体フレームでノイズを乗せる厳密経路
    (simulate_observation_attitude / run_attitude.py, §14) を使うこと。本ヘルパは位置レベルの
    感度を手早く見る用途に留める。既定 enable=False では何も変えない (後方互換)。
    """
    z = np.asarray(z_world_seq, dtype=float)
    if not enable:
        return z
    one = (z.ndim == 1)              # 単一観測 (3,) も受ける (静止点シナリオ用)。n=1 として処理。
    if one:
        z = z[None, :]
    from src.truth import wave_attitude_sequence              # 親機姿勢の真値 (§14.1)
    from src.attitude import (euler_to_matrix, complementary_filter,
                              correct_observation_sequence, body_bearing_to_world)
    if dt is None:
        from src.config import ATT_DT
        dt = ATT_DT
    n = len(z)
    e_true = wave_attitude_sequence(n, dt=dt, seed=seed + 700, **(wave or {}))
    R_true = np.array([euler_to_matrix(*ev) for ev in e_true])
    # ワールド角度 -> 機体角度 (親機が R だけ回ると、ワールド方向は機体では R^T だけ回って見える)
    z_body = np.array([body_bearing_to_world(z[k], R_true[k].T) for k in range(n)])
    if not imu_correct:
        out = z_body                                           # naive: 補正しない
    else:
        sig = simulate_imu_signals(R_true, dt=dt, seed=seed + 800, gyro_sigma=gyro_sigma,
                                   gyro_bias=gyro_bias, acc_sigma=acc_sigma,
                                   mag_sigma=mag_sigma, gravity=gravity)
        R_est = complementary_filter(sig["gyro"], sig["acc"], sig["mag"], dt=dt,
                                     alpha=filter_alpha)
        out = correct_observation_sequence(z_body, R_est)      # IMU推定姿勢でワールドへ補正
    return out[0] if one else out


def apply_attitude_error_config(z_world_seq, seed):
    """config [attitude] の設定で apply_attitude_error を呼ぶ薄いラッパ (各シナリオ用)。

    [attitude].as_error=False (既定) では z をそのまま返す = 全シナリオで従来結果と一致。
    True にすると wave/IMU パラメータと imu_correct を config から読んで波動揺誤差を重ねる。
    """
    from src import config
    return apply_attitude_error(
        z_world_seq, seed, enable=config.ATT_AS_ERROR, imu_correct=config.ATT_IMU_CORRECT,
        wave=config.ATT_WAVE, dt=config.ATT_DT, filter_alpha=config.ATT_FILTER_ALPHA,
        **config.ATT_IMU_KW)


# ----------------------------------------------------------------------------
# 波による親機動揺を SBL アンカーアレイの回転として反映する (MATH_SPEC §13.5/§14)
#
# SBL の4トランスデューサは親機ピボット (p_parent) から**オフセット**して付くので、親機が波で
# 姿勢 R(t) に揺れるとアレイ全体がワールドで回り、各レンジ d_i=||p_child - A_i(t)|| が変わる。
# (単一距離フォールバック §11 はピボット上の1点なので回転不変 = 波動揺に不感。SBL はオフセット
#  ぶん不感ではない、という非対称がここの肝。)
#
# 本ヘルパは「真値レンジ生成に使う真の回転アンカー列」と「推定が使うアンカー」を返す:
#   - imu_correct=False (naive): 推定は公称(level)アンカーのまま → 波動揺の系統誤差が丸ごと残る
#   - imu_correct=True         : 親機 IMU 相補フィルタの推定姿勢 R_est でアンカーを回す
#                                → 姿勢推定残差ぶんだけ誤差が残る (apply_attitude_error と対称)
# enable=False (既定) では (anchors, anchors) を返す = 従来結果と完全一致 (後方互換)。
# truth (R_true) は本ヘルパ内のみで使い、estimator には R_est 由来のアンカーしか渡さない
# (MBD 分離は保たれる)。
# ----------------------------------------------------------------------------
def sbl_attitude_anchors(anchors, n, seed, *, enable, imu_correct=True, wave=None,
                         dt=None, p_parent=None, gyro_sigma=0.0, gyro_bias=0.0,
                         acc_sigma=0.0, mag_sigma=0.0, gravity=9.80665, filter_alpha=0.98):
    """親機の波動揺で回る SBL アンカー列を返す (MATH_SPEC §13.5/§14)。

    anchors : (M,3) 公称 (level) のアンカー配置 [m]。
    n       : 時刻数 (軌道点数)。
    戻り値  : (anchors_true (n,M,3), anchors_est ((M,3) naive / (n,M,3) imu_correct))
              anchors_true は simulate_sbl_range_sequence に渡す**真値**アンカー、
              anchors_est は estimate_trajectory_sbl に渡す推定側アンカー。
    enable=False なら (anchors, anchors) を返す = 従来一致。
    """
    anchors = np.asarray(anchors, dtype=float)
    if not enable:
        return anchors, anchors
    p_parent = np.zeros(3) if p_parent is None else np.asarray(p_parent, dtype=float)
    from src.truth import wave_attitude_sequence              # 親機姿勢の真値 (§14.1)
    from src.attitude import euler_to_matrix, complementary_filter
    if dt is None:
        from src.config import ATT_DT
        dt = ATT_DT
    off = anchors - p_parent                                  # 機体フレームのアンカーオフセット
    e_true = wave_attitude_sequence(n, dt=dt, seed=seed + 700, **(wave or {}))
    R_true = np.array([euler_to_matrix(*ev) for ev in e_true])
    anchors_true = p_parent + np.einsum("kij,mj->kmi", R_true, off)   # (n,M,3) 波で回った真アンカー
    if not imu_correct:
        return anchors_true, anchors                          # naive: 推定は公称(level)アンカー
    sig = simulate_imu_signals(R_true, dt=dt, seed=seed + 800, gyro_sigma=gyro_sigma,
                               gyro_bias=gyro_bias, acc_sigma=acc_sigma,
                               mag_sigma=mag_sigma, gravity=gravity)
    R_est = complementary_filter(sig["gyro"], sig["acc"], sig["mag"], dt=dt, alpha=filter_alpha)
    anchors_est = p_parent + np.einsum("kij,mj->kmi", R_est, off)     # IMU推定姿勢で回したアンカー
    return anchors_true, anchors_est


def sbl_attitude_anchors_config(anchors, n, seed, p_parent=None):
    """config [attitude] の設定で sbl_attitude_anchors を呼ぶ薄いラッパ (SBL シナリオ用)。

    [attitude].as_error=False (既定) では (anchors, anchors) を返す = 従来結果と一致。
    True にすると wave/IMU パラメータと imu_correct を config から読み、波動揺で回るアンカーを返す。
    """
    from src import config
    return sbl_attitude_anchors(
        anchors, n, seed, enable=config.ATT_AS_ERROR, imu_correct=config.ATT_IMU_CORRECT,
        wave=config.ATT_WAVE, dt=config.ATT_DT, p_parent=p_parent,
        filter_alpha=config.ATT_FILTER_ALPHA, **config.ATT_IMU_KW)
