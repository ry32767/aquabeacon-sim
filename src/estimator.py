"""estimator.py — ③ 位置推定 (MATH_SPEC §3, §4)。

真値を見てはならない層。入力は観測値 (d, theta, phi) と既知の親機位置・ノイズ
パラメータのみ。重み付き最小二乗で子機位置を推定する。

このモジュールは truth を import しない (MBD 層分離。test_separation が強制)。
観測モデル h は sensors の純粋な変換関数 (forward/inverse_observation) を使う。
"""
import numpy as np
from scipy.optimize import least_squares

from src.sensors import forward_observation, inverse_observation, relative_vector


def wrap_angle(a):
    """角度差を (-pi, pi] に正規化する (MATH_SPEC §3.2)。±pi 境界の不連続を防ぐ。"""
    return np.arctan2(np.sin(a), np.cos(a))


def h(x_state, p_parent=None, observe_from_parent=True):
    """観測モデル: 推定状態 x=(x,y,z) -> 予測観測 (d_hat, theta_hat, phi_hat)  (MATH_SPEC §3.1)。"""
    if p_parent is None:
        p_parent = np.zeros(3)
    v = relative_vector(x_state, p_parent, observe_from_parent)
    return forward_observation(v)


def residual(x_state, z_meas, p_parent=None, observe_from_parent=True):
    """残差 r(x) = z - h(x)。角度成分は wrap_angle で正規化する (MATH_SPEC §3.2)。"""
    d_hat, th_hat, ph_hat = h(x_state, p_parent, observe_from_parent)
    d, th, ph = z_meas
    return np.array([
        d - d_hat,
        wrap_angle(th - th_hat),
        wrap_angle(ph - ph_hat),
    ])


def weight_matrix(sigma_dist, sigma_az, sigma_el):
    """重み行列 W = diag(1/sigma^2)  (MATH_SPEC §4.1)。精度の良い観測ほど重い。"""
    return np.diag([1.0 / sigma_dist**2, 1.0 / sigma_az**2, 1.0 / sigma_el**2])


def predicted_depth(x_state):
    """深度センサの観測モデル: 状態 x=(x,y,z) -> 予測深度 = -z [m, 下が正] (MATH_SPEC §10)。"""
    return -np.asarray(x_state, dtype=float)[2]


def depth_residual(x_state, z_depth):
    """深度残差 r_depth = z_depth - (-z) = z_depth + z  (MATH_SPEC §10)。"""
    return z_depth - predicted_depth(x_state)


# ロバスト損失 (MATH_SPEC §4.4)。'linear' は従来の純 L2 (最小二乗)。
# それ以外は M推定 (外れ値の影響を抑える)。残差は sqrt(W) で σ 正規化済みなので、
# f_scale は「σ の何倍までを内れ値とみなすか」を表す (Huber 既定 1.345σ)。
_ROBUST_LOSSES = ("linear", "soft_l1", "huber", "cauchy", "arctan")


def _solve_least_squares(fun, x0, loss="linear", f_scale=1.345):
    """残差関数 fun を最小化する。loss='linear' なら従来の LM、それ以外は TRF + ロバスト損失。

    scipy の least_squares はロバスト損失 (huber 等) を method='trf'/'dogbox' でのみ
    サポートする (LM は純 L2 のみ)。loss='linear' のときは従来挙動を完全に保つため LM を使う。

    ロバスト損失 (特に cauchy のような redescending 系) は初期値依存が強く、外れ値で
    初期値が汚れていると悪い極小に落ちる。これを避けるため、まず純 L2 (LM) で温めた解を
    初期値にして robust を精緻化する 2段解法にする (warm start)。loss='linear' は1回解くだけ。
    """
    if loss not in _ROBUST_LOSSES:
        raise ValueError(f"loss は {_ROBUST_LOSSES} のいずれか (受領: {loss!r})")
    if loss == "linear":
        return least_squares(fun, x0, method="lm")
    warm = least_squares(fun, x0, method="lm")            # L2 で温める
    return least_squares(fun, warm.x, method="trf", loss=loss, f_scale=f_scale)


def estimate_position(z_meas, sigma, p_parent=None,
                      observe_from_parent=True, x0=None,
                      loss="linear", f_scale=1.345,
                      z_depth=None, sigma_depth=None):
    """重み付き(ロバスト)最小二乗で子機位置を推定する (MATH_SPEC §4.2, §4.4, §10)。

    z_meas : 観測 (d, theta, phi)
    sigma  : (sigma_dist [m], sigma_az [rad], sigma_el [rad])
    p_parent: 親機位置 (既知)。None なら原点。
    x0     : 初期値。None なら §2 の逆変換で算出する。
    loss   : 'linear' (純L2, 既定) / 'huber' / 'cauchy' / 'soft_l1' / 'arctan'。
    f_scale: ロバスト損失の内れ値しきい値 (σ単位, 既定 1.345)。loss='linear' では無視。
    z_depth: 深度センサ観測 [m, 下が正] (MATH_SPEC §10)。None で未使用。
    sigma_depth: 深度ノイズ [m]。z_depth を使うとき必須。
    戻り値 : 推定位置 x_hat=(x,y,z)

    least_squares は残差ベクトルを最小化するので、重み W=diag(1/sigma^2) の平方根
    sqrt(W)=diag(1/sigma) を残差に掛けて渡す (r^T W r と等価)。

    冗長性とロバスト: 単時刻で観測 (d,θ,φ) のみだと観測3・未知数3で**冗長性が無い**ため、
    ロバスト損失でも外れ値を棄却できない。**深度センサ z_depth を加えると観測4・未知数3で
    冗長性が生まれ、単時刻でもロバスト推定が外れ値を識別・減衰できる** (§10)。深度は鉛直 z を
    距離・濁りに依存せず直接拘束するので、光学が苦手な深い/濁った水で特に有効。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    if z_depth is not None and sigma_depth is None:
        raise ValueError("z_depth を使うときは sigma_depth が必要です")
    sd, sa, se = sigma
    sqrtW = np.array([1.0 / sd, 1.0 / sa, 1.0 / se])
    sqrtW_depth = None if z_depth is None else 1.0 / sigma_depth

    if x0 is None:
        v0 = inverse_observation(*z_meas)
        x0 = (p_parent + v0) if observe_from_parent else (p_parent - v0)

    def weighted_residual(x_state):
        r = sqrtW * residual(x_state, z_meas, p_parent, observe_from_parent)
        if z_depth is not None:
            r = np.append(r, sqrtW_depth * depth_residual(x_state, z_depth))
        return r

    sol = _solve_least_squares(weighted_residual, x0, loss=loss, f_scale=f_scale)
    return sol.x


def estimate_trajectory(z_seq, sigma_obs, imu_deltas=None, sigma_imu=None,
                        p_parent=None, observe_from_parent=True, x0=None,
                        loss="linear", f_scale=1.345,
                        z_depth_seq=None, sigma_depth=None, use_angles=True):
    """複数時刻の観測 (+ IMU 拘束 + 深度) から子機軌道を一括推定する (MATH_SPEC §5, §4.4, §10, §11)。

    z_seq     : (n,3) 各時刻の観測 (d, theta, phi)
    sigma_obs : (sigma_dist, sigma_az, sigma_el) 観測ノイズ
    imu_deltas: (n-1,3) 時刻間変位 delta_p の IMU 観測。None なら IMU 拘束なし。
    sigma_imu : IMU 変位ノイズ (スカラ or (3,))。imu_deltas を使うとき必須。
    x0        : 初期軌道 (n,3)。None なら各時刻を §2 逆変換で初期化する。
    loss      : 'linear' (純L2, 既定) / 'huber' / 'cauchy' / 'soft_l1' / 'arctan'。
    f_scale   : ロバスト損失の内れ値しきい値 (σ単位, 既定 1.345)。loss='linear' では無視。
    z_depth_seq: (n,) 各時刻の深度センサ観測 [m, 下が正] (MATH_SPEC §10)。None で未使用。
    sigma_depth: 深度ノイズ [m]。z_depth_seq を使うとき必須。
    use_angles: True で観測 (d,θ,φ) 全部を使う。False なら**距離 d のみ**使い方位/仰角を
                捨てる (光学追跡なしのフォールバック §11)。False のときは方位が不可観測なので
                IMU と深度の併用が前提。逆変換初期化が使えないため x0 を与えること。
    戻り値    : 推定軌道 X_hat (n,3)

    外れ値対策 (MATH_SPEC §4.4): IMU 拘束が時刻間を繋ぎ冗長性を作るので、ある時刻の
    観測が外れ値 (ライト見失い・音響マルチパス) でも、loss='huber'/'cauchy' なら
    その残差を自動で減衰し、軌道全体の破綻を防ぐ。深度センサ (§10) は各時刻の鉛直 z を
    直接拘束し、深い/濁った水で光学仰角が劣化しても軌道の z を安定させる。

    目的関数 (MATH_SPEC §5.3):
        X_hat = argmin_X  Σ_k ||r_obs_k||^2_{W_obs} + Σ_k ||r_imu_k||^2_{W_imu}
        r_obs_k = z_k - h(x_k)              (3節)
        r_imu_k = (x_{k+1} - x_k) - delta_p_imu_k
    least_squares に全残差を連結したベクトルを渡して最小化する。
    入力は観測値のみ。truth は一切参照しない (MBD)。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    z_seq = np.asarray(z_seq, dtype=float)
    n = len(z_seq)

    sd, sa, se = sigma_obs
    sqrtW_obs = np.array([1.0 / sd, 1.0 / sa, 1.0 / se])

    use_imu = imu_deltas is not None
    if use_imu:
        if sigma_imu is None:
            raise ValueError("imu_deltas を使うときは sigma_imu が必要です")
        imu_deltas = np.asarray(imu_deltas, dtype=float)
        sqrtW_imu = 1.0 / np.broadcast_to(np.asarray(sigma_imu, dtype=float), (3,))

    use_depth = z_depth_seq is not None
    if use_depth:
        if sigma_depth is None:
            raise ValueError("z_depth_seq を使うときは sigma_depth が必要です")
        z_depth_seq = np.asarray(z_depth_seq, dtype=float).reshape(n)
        sqrtW_depth = 1.0 / sigma_depth

    # 初期値: 各時刻を逆変換で (use_angles=False では角度が無く逆変換不可 -> x0 必須)
    if x0 is None:
        if not use_angles:
            raise ValueError("use_angles=False のときは x0 (初期軌道) を与えてください "
                             "(方位が不可観測なため逆変換初期化が使えない, §11)")
        x0 = np.empty((n, 3))
        for k in range(n):
            v0 = inverse_observation(*z_seq[k])
            x0[k] = (p_parent + v0) if observe_from_parent else (p_parent - v0)
    x0 = np.asarray(x0, dtype=float).reshape(n, 3)

    def stacked_residual(x_flat):
        X = x_flat.reshape(n, 3)
        parts = []
        for k in range(n):       # 観測残差 (use_angles=False なら距離 d のみ, §11)
            r_obs = residual(X[k], z_seq[k], p_parent, observe_from_parent)
            if use_angles:
                parts.append(sqrtW_obs * r_obs)
            else:
                parts.append(np.array([sqrtW_obs[0] * r_obs[0]]))
        if use_imu:              # IMU 拘束残差
            for k in range(n - 1):
                parts.append(sqrtW_imu * ((X[k + 1] - X[k]) - imu_deltas[k]))
        if use_depth:            # 深度センサ残差 (各時刻スカラ, MATH_SPEC §10)
            for k in range(n):
                parts.append(np.array([sqrtW_depth *
                                       depth_residual(X[k], z_depth_seq[k])]))
        return np.concatenate(parts)

    sol = _solve_least_squares(stacked_residual, x0.ravel(), loss=loss,
                               f_scale=f_scale)
    return sol.x.reshape(n, 3)


def _acoustic_inertial_cost(X, range_seq, sigma_dist, imu_deltas, sigma_imu,
                            depth_seq, sigma_depth, p_parent):
    """光学なし推定の重み付き残差二乗和 (多スタート選択用, MATH_SPEC §11)。"""
    n = len(X)
    si = np.broadcast_to(np.asarray(sigma_imu, dtype=float), (3,))
    c = 0.0
    for k in range(n):
        d_hat = np.linalg.norm(relative_vector(X[k], p_parent))
        c += ((range_seq[k] - d_hat) / sigma_dist) ** 2
        c += (depth_residual(X[k], depth_seq[k]) / sigma_depth) ** 2
    for k in range(n - 1):
        c += float(np.sum((((X[k + 1] - X[k]) - imu_deltas[k]) / si) ** 2))
    return c


def estimate_trajectory_acoustic_inertial(range_seq, sigma_dist, imu_deltas, sigma_imu,
                                          depth_seq, sigma_depth, p_parent=None,
                                          observe_from_parent=True,
                                          n_azimuth_starts=12,
                                          loss="linear", f_scale=1.345):
    """光学追跡なし (距離 + IMU + 深度) で子機軌道を推定する (MATH_SPEC §11)。

    濁り水でビーコンを見失う等で方位/仰角が得られない場合のフォールバック。
      - 距離 (音響)     : 各時刻 ||x_k - p_parent|| = d_k   (球面拘束)
      - 深度 (圧力)     : -z_k = depth_k                    (鉛直を直接拘束, §10)
      - IMU (世界座標変位): x_{k+1} - x_k = Δp_k             (時刻間を繋ぐ, §5)

    観測可能性 (§11): 単時刻は距離+深度の2拘束で方位が不可観測 (円)。IMU で軌道形状を
    繋ぐと、距離トリラテレーション+深度で軌道が一意に定まる (非退化運動が前提)。方位の
    局所解 (基底) を避けるため、開始方位を n_azimuth_starts 通りグリッド探索し最良解を返す。

    range_seq : (n,) 音響距離 [m]
    imu_deltas: (n-1,3) 世界座標の時刻間変位 [m] (向き既知=方位基準ありを仮定)
    depth_seq : (n,) 深度 [m, 下が正]
    戻り値    : 推定軌道 X_hat (n,3)。truth は参照しない (MBD)。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    range_seq = np.asarray(range_seq, dtype=float)
    depth_seq = np.asarray(depth_seq, dtype=float)
    imu_deltas = np.asarray(imu_deltas, dtype=float)
    n = len(range_seq)

    z_seq = np.zeros((n, 3))
    z_seq[:, 0] = range_seq                          # 角度成分は use_angles=False で無視
    sigma_obs = (sigma_dist, 1.0, 1.0)               # 角度σはダミー

    # IMU を積分した世界座標の相対形状 (向きは既知, 開始点だけ未知)
    shape = np.vstack([np.zeros(3), np.cumsum(imu_deltas, axis=0)])
    z0 = -depth_seq[0]
    rho0 = np.sqrt(max(range_seq[0] ** 2 - z0 ** 2, 0.0))   # 開始点の水平半径

    best, best_cost = None, np.inf
    for j in range(n_azimuth_starts):
        a = 2.0 * np.pi * j / n_azimuth_starts
        start = np.array([rho0 * np.cos(a), rho0 * np.sin(a), z0])
        x0 = start + shape
        x0[:, 2] = -depth_seq                        # z は深度で初期化
        est = estimate_trajectory(
            z_seq, sigma_obs, imu_deltas=imu_deltas, sigma_imu=sigma_imu,
            p_parent=p_parent, observe_from_parent=observe_from_parent, x0=x0,
            loss=loss, f_scale=f_scale, z_depth_seq=depth_seq,
            sigma_depth=sigma_depth, use_angles=False)
        c = _acoustic_inertial_cost(est, range_seq, sigma_dist, imu_deltas,
                                    sigma_imu, depth_seq, sigma_depth, p_parent)
        if c < best_cost:
            best_cost, best = c, est
    return best
