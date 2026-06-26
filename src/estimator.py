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


# ----------------------------------------------------------------------------
# 解析ヤコビアン (MATH_SPEC §4.3)。観測モデル h=(d,θ,φ) の状態 x=(x,y,z) に関する
# 偏微分。数値ヤコビアンより速く・正確で、収束と共分散 (§4.5) の双方に使う。
# truth を参照しない (位置を引数で受けるだけ。MBD 分離は保たれる)。
# ----------------------------------------------------------------------------
def _numeric_obs_jacobian(x_state, p_parent, observe_from_parent, eps=1e-7):
    """h の 3x3 数値ヤコビアン (中心差分)。rho≈0 特異近傍のフォールバック (§4.3)。"""
    x_state = np.asarray(x_state, dtype=float)
    J = np.empty((3, 3))
    for j in range(3):
        dx = np.zeros(3)
        dx[j] = eps
        hp = h(x_state + dx, p_parent, observe_from_parent)
        hm = h(x_state - dx, p_parent, observe_from_parent)
        diff = hp - hm
        diff[1] = wrap_angle(diff[1])      # 角度は ±pi 境界をまたがないよう正規化
        diff[2] = wrap_angle(diff[2])
        J[:, j] = diff / (2.0 * eps)
    return J


def observation_jacobian(x_state, p_parent=None, observe_from_parent=True, eps=1e-9):
    """観測モデル h=(d,θ,φ) の 3x3 解析ヤコビアン ∂h/∂x (MATH_SPEC §4.3)。

    相対ベクトル v=(vx,vy,vz)=±(x-p_parent), d=||v||, rho=hypot(vx,vy) として:
        ∂d/∂v     = (vx/d,  vy/d,  vz/d)
        ∂theta/∂v = (-vy/rho^2,  vx/rho^2,  0)
        ∂phi/∂v   = (-vx*vz/(d^2*rho),  -vy*vz/(d^2*rho),  rho/d^2)
    ∂v/∂x = +I (observe_from_parent=True) / -I (False) なので ∂h/∂x = ±∂h/∂v。
    rho≈0 (真上・真下) では theta/phi の微分が特異なので数値微分にフォールバックする。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    v = relative_vector(x_state, p_parent, observe_from_parent)
    vx, vy, vz = v
    d2 = vx * vx + vy * vy + vz * vz
    rho2 = vx * vx + vy * vy
    d = np.sqrt(d2)
    rho = np.sqrt(rho2)
    if d < eps or rho < eps:               # 特異近傍は数値ヤコビアン (§4.3)
        return _numeric_obs_jacobian(x_state, p_parent, observe_from_parent)
    dh_dv = np.array([
        [vx / d,             vy / d,             vz / d],
        [-vy / rho2,         vx / rho2,          0.0],
        [-vx * vz / (d2 * rho), -vy * vz / (d2 * rho), rho / d2],
    ])
    sign = 1.0 if observe_from_parent else -1.0
    return sign * dh_dv                     # ∂h/∂x = ∂h/∂v · (±I)


# ----------------------------------------------------------------------------
# 解の不確かさ: 共分散・GDOP (MATH_SPEC §4.5)。重み付き最小二乗の解の共分散は
#   Cov = (JᵀWJ)⁻¹           (J = ∂h/∂x, W = diag(1/σ²))
# で与えられる (ガウス・ML 仮定での CRLB に一致)。観測幾何 (位置) と σ のみで決まり、
# truth も観測ノイズの実現値も要らない。位置を引数で受けるので MBD 分離を保つ
# (真値を入れれば CRLB、推定値を入れれば推定の事後共分散。CRLB は evaluation 側で呼ぶ)。
# ----------------------------------------------------------------------------
def position_covariance(x_state, sigma, p_parent=None, observe_from_parent=True,
                        with_depth=False, sigma_depth=None, rcond=1e-12):
    """単時刻測位の解の共分散 Cov=(JᵀWJ)⁻¹ [m²] (3x3) を返す (MATH_SPEC §4.5)。

    x_state : 共分散を評価する位置 (真値→CRLB / 推定値→事後共分散)
    sigma   : (σ_dist, σ_az[rad], σ_el[rad])
    with_depth/sigma_depth: 深度センサ (§10) を融合した場合の冗長拘束を加える。
    rho≈0 で情報行列が特異になりうるため pinv を使う (rcond で打ち切り)。
    """
    sd, sa, se = sigma
    sqrtW = np.array([1.0 / sd, 1.0 / sa, 1.0 / se])
    Jh = observation_jacobian(x_state, p_parent, observe_from_parent)
    Jw = sqrtW[:, None] * Jh                       # sqrt(W) 正規化した観測ヤコビアン
    if with_depth:
        if sigma_depth is None:
            raise ValueError("with_depth=True のときは sigma_depth が必要です")
        # 深度残差 r=z_depth+z の ∂/∂x=(0,0,1)、重み 1/σ_depth
        Jw = np.vstack([Jw, np.array([[0.0, 0.0, 1.0 / sigma_depth]])])
    F = Jw.T @ Jw                                  # Fisher 情報 = JᵀWJ
    return np.linalg.pinv(F, rcond=rcond)


def gdop(cov):
    """幾何希釈 GDOP = sqrt(trace(Cov)) [m] (位置共分散の RMS 半径, MATH_SPEC §4.5)。

    cov が (3,3) なら単時刻、(n,3,3) なら各時刻の GDOP 配列 (n,) を返す。
    σ で割れば無次元 DOP になる (呼び出し側で実施)。
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 2:
        return float(np.sqrt(np.trace(cov)))
    return np.sqrt(np.trace(cov, axis1=1, axis2=2))


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


def _solve_least_squares(fun, x0, loss="linear", f_scale=1.345, jac=None):
    """残差関数 fun を最小化する。loss='linear' なら従来の LM、それ以外は TRF + ロバスト損失。

    scipy の least_squares はロバスト損失 (huber 等) を method='trf'/'dogbox' でのみ
    サポートする (LM は純 L2 のみ)。loss='linear' のときは従来挙動を完全に保つため LM を使う。

    jac: 解析ヤコビアンのコールバック (MATH_SPEC §4.3)。None なら scipy の数値ヤコビアン
         (従来挙動)。与えると収束が速く・正確になる (特に軌道・SBL・光学なしの大規模解で顕著)。
         well-posed 問題では収束解は数値ヤコビアン版と一致する (atol≪要求精度)。

    ロバスト損失 (特に cauchy のような redescending 系) は初期値依存が強く、外れ値で
    初期値が汚れていると悪い極小に落ちる。これを避けるため、まず純 L2 (LM) で温めた解を
    初期値にして robust を精緻化する 2段解法にする (warm start)。loss='linear' は1回解くだけ。
    非収束を検出したら警告する (黙って誤った解を返さない, MATH_SPEC §4.5 注)。
    """
    if loss not in _ROBUST_LOSSES:
        raise ValueError(f"loss は {_ROBUST_LOSSES} のいずれか (受領: {loss!r})")
    jac_arg = jac if jac is not None else "2-point"
    if loss == "linear":
        sol = least_squares(fun, x0, method="lm", jac=jac_arg)
        _warn_if_not_converged(sol, loss)
        return sol
    warm = least_squares(fun, x0, method="lm", jac=jac_arg)   # L2 で温める
    x_seed = warm.x if warm.success else np.asarray(x0, dtype=float)
    if not warm.success:                                  # 温め解が壊れたら x0 から (汚染防止)
        _warn_if_not_converged(warm, "linear(warm-start)")
    sol = least_squares(fun, x_seed, method="trf", loss=loss, f_scale=f_scale,
                        jac=jac_arg)
    _warn_if_not_converged(sol, loss)
    return sol


def _warn_if_not_converged(sol, loss):
    """least_squares の結果が非収束 (status<=0) または非有限コストなら警告する (§4.5)。"""
    if (not getattr(sol, "success", True)) or (not np.isfinite(getattr(sol, "cost", 0.0))):
        import warnings
        warnings.warn(
            f"least_squares が収束しませんでした (loss={loss!r}, "
            f"status={getattr(sol, 'status', '?')}, cost={getattr(sol, 'cost', '?')}). "
            f"返り値は最良努力解です。",
            RuntimeWarning, stacklevel=3)


def estimate_position(z_meas, sigma, p_parent=None,
                      observe_from_parent=True, x0=None,
                      loss="linear", f_scale=1.345,
                      z_depth=None, sigma_depth=None, return_cov=False):
    """重み付き(ロバスト)最小二乗で子機位置を推定する (MATH_SPEC §4.2, §4.4, §10)。

    z_meas : 観測 (d, theta, phi)
    sigma  : (sigma_dist [m], sigma_az [rad], sigma_el [rad])
    p_parent: 親機位置 (既知)。None なら原点。
    x0     : 初期値。None なら §2 の逆変換で算出する。
    loss   : 'linear' (純L2, 既定) / 'huber' / 'cauchy' / 'soft_l1' / 'arctan'。
    f_scale: ロバスト損失の内れ値しきい値 (σ単位, 既定 1.345)。loss='linear' では無視。
    z_depth: 深度センサ観測 [m, 下が正] (MATH_SPEC §10)。None で未使用。
    sigma_depth: 深度ノイズ [m]。z_depth を使うとき必須。
    return_cov: True なら (x_hat, cov) を返す。cov は解の 3x3 共分散 (JᵀWJ)⁻¹ [m²]
                (MATH_SPEC §4.5)。既定 False では従来どおり x_hat のみ (後方互換)。
    戻り値 : 推定位置 x_hat=(x,y,z) (return_cov=True なら (x_hat, cov))

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

    def weighted_jac(x_state):              # 残差の解析ヤコビアン (MATH_SPEC §4.3)
        Jh = observation_jacobian(x_state, p_parent, observe_from_parent)
        J = -sqrtW[:, None] * Jh            # ∂(z-h)/∂x = -∂h/∂x
        if z_depth is not None:             # 深度残差 r=z_depth+z の ∂/∂x=(0,0,1)
            J = np.vstack([J, np.array([[0.0, 0.0, sqrtW_depth]])])
        return J

    sol = _solve_least_squares(weighted_residual, x0, loss=loss, f_scale=f_scale,
                               jac=weighted_jac)
    if return_cov:
        cov = position_covariance(sol.x, sigma, p_parent, observe_from_parent,
                                  with_depth=(z_depth is not None),
                                  sigma_depth=sigma_depth)
        return sol.x, cov
    return sol.x


def estimate_trajectory(z_seq, sigma_obs, imu_deltas=None, sigma_imu=None,
                        p_parent=None, observe_from_parent=True, x0=None,
                        loss="linear", f_scale=1.345,
                        z_depth_seq=None, sigma_depth=None, use_angles=True,
                        angle_mask=None, return_cov=False):
    """複数時刻の観測 (+ IMU 拘束 + 深度) から子機軌道を一括推定する (MATH_SPEC §5, §4.4, §10, §11, §12)。

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
    angle_mask: (n,) bool。各時刻で方位/仰角を使うか個別指定 (自動切替 §12)。True の時刻は
                (d,θ,φ)、False の時刻は距離 d のみ。None なら use_angles を全時刻に適用する。
                光学が一部時刻だけ使える混在ケースに用いる (x0 を与えること)。
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

    # 観測重み: sigma_obs は (3,) 全時刻共通 (従来) か (n,3) 時刻別 (距離/濁り依存の
    # ヘテロスケダスティック WLS, MATH_SPEC §5.4)。(3,) は broadcast で従来と完全一致。
    sigma_obs = np.asarray(sigma_obs, dtype=float)
    if sigma_obs.shape == (3,):
        sqrtW_obs_all = np.broadcast_to(1.0 / sigma_obs, (n, 3))
    elif sigma_obs.shape == (n, 3):
        sqrtW_obs_all = 1.0 / sigma_obs
    else:
        raise ValueError(f"sigma_obs は (3,) か (n,3) 形状: 受領 {sigma_obs.shape}")

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

    # 各時刻で角度を使うかのマスク (angle_mask 優先, 無ければ use_angles を全時刻に)
    if angle_mask is None:
        angle_mask = np.full(n, bool(use_angles))
    else:
        angle_mask = np.asarray(angle_mask, dtype=bool).reshape(n)
    all_angles = bool(angle_mask.all())

    # 初期値: 各時刻を逆変換で (角度の無い時刻があると逆変換不可 -> x0 必須)
    if x0 is None:
        if not all_angles:
            raise ValueError("角度を使わない時刻があるときは x0 (初期軌道) を与えてください "
                             "(方位が不可観測なため逆変換初期化が使えない, §11/§12)")
        x0 = np.empty((n, 3))
        for k in range(n):
            v0 = inverse_observation(*z_seq[k])
            x0[k] = (p_parent + v0) if observe_from_parent else (p_parent - v0)
    x0 = np.asarray(x0, dtype=float).reshape(n, 3)

    def stacked_residual(x_flat):
        X = x_flat.reshape(n, 3)
        parts = []
        for k in range(n):       # 観測残差 (角度マスク False の時刻は距離 d のみ, §11/§12)
            r_obs = residual(X[k], z_seq[k], p_parent, observe_from_parent)
            if angle_mask[k]:
                parts.append(sqrtW_obs_all[k] * r_obs)
            else:
                parts.append(np.array([sqrtW_obs_all[k][0] * r_obs[0]]))
        if use_imu:              # IMU 拘束残差
            for k in range(n - 1):
                parts.append(sqrtW_imu * ((X[k + 1] - X[k]) - imu_deltas[k]))
        if use_depth:            # 深度センサ残差 (各時刻スカラ, MATH_SPEC §10)
            for k in range(n):
                parts.append(np.array([sqrtW_depth *
                                       depth_residual(X[k], z_depth_seq[k])]))
        return np.concatenate(parts)

    # 残差行数 M (stacked_residual と同じ順序・本数)。解析ヤコビアン構築に使う。
    n_obs_rows = int(np.where(angle_mask, 3, 1).sum())
    M = n_obs_rows + (3 * (n - 1) if use_imu else 0) + (n if use_depth else 0)

    def stacked_jac(x_flat):     # ブロック疎な解析ヤコビアン (MATH_SPEC §4.3, §5)
        X = x_flat.reshape(n, 3)
        J = np.zeros((M, 3 * n))
        r = 0
        for k in range(n):
            Jh = observation_jacobian(X[k], p_parent, observe_from_parent)
            sk = sqrtW_obs_all[k]
            if angle_mask[k]:
                J[r:r + 3, 3 * k:3 * k + 3] = -sk[:, None] * Jh
                r += 3
            else:
                J[r, 3 * k:3 * k + 3] = -sk[0] * Jh[0]
                r += 1
        if use_imu:              # r_imu=(x_{k+1}-x_k)-Δp: ∂/∂x_k=-I, ∂/∂x_{k+1}=+I
            for k in range(n - 1):
                for a in range(3):
                    J[r + a, 3 * k + a] = -sqrtW_imu[a]
                    J[r + a, 3 * (k + 1) + a] = sqrtW_imu[a]
                r += 3
        if use_depth:            # r_depth=z_depth+z: ∂/∂z=+1
            for k in range(n):
                J[r, 3 * k + 2] = sqrtW_depth
                r += 1
        return J

    sol = _solve_least_squares(stacked_residual, x0.ravel(), loss=loss,
                               f_scale=f_scale, jac=stacked_jac)
    X_hat = sol.x.reshape(n, 3)
    if return_cov:               # 各時刻の 3x3 共分散ブロック (MATH_SPEC §4.5)
        Jw = stacked_jac(sol.x)
        cov_full = np.linalg.pinv(Jw.T @ Jw)
        cov = np.array([cov_full[3 * k:3 * k + 3, 3 * k:3 * k + 3] for k in range(n)])
        return X_hat, cov
    return X_hat


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

    # 方位の可観測性 (§11.2.1): 水平 IMU 運動が2次元に広がれば方位は1スタートで解ける。
    # ほぼ直進だと運動線に対する左右の鏡像が等コストになり、グリッド探索でも一意に決まらない
    # (ノイズ下では誤った basin が低コストになり得る)。退化時は警告し、非退化時は無駄な
    # 多スタートを省く (解は多スタートと一致。analytic Jacobian (§4.3) で1解は十分高速)。
    H = imu_deltas[:, :2]
    sv = np.linalg.svd(H, compute_uv=False) if len(H) else np.array([0.0])
    richness = float(sv[-1] / sv[0]) if (len(sv) >= 2 and sv[0] > 1e-12) else 0.0
    if richness < 0.05:                              # ほぼ直進 -> 鏡像不定 (§11.2.1)
        import warnings
        warnings.warn(
            "水平運動がほぼ直進のため方位が不可観測です (鏡像不定, MATH_SPEC §11.2.1)。"
            "推定は運動線に対する左右いずれかの解を返し得ます。方位アンカー (SBL §13) か"
            "旋回を含む運動を推奨します。", RuntimeWarning, stacklevel=2)
        starts = n_azimuth_starts
    else:                                            # 非退化: 1スタートで大域解に到達
        starts = 1 if richness > 0.3 else n_azimuth_starts

    best, best_cost = None, np.inf
    for j in range(starts):
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
        if best is None or c < best_cost:
            best_cost, best = c, est
    if best is None or not np.isfinite(best_cost):   # 全スタート失敗は黙らず例外 (§4.5)
        raise RuntimeError("光学なし軌道推定が全ての開始方位で失敗しました "
                           "(非収束/非有限コスト)。観測・初期値・退化運動を確認してください。")
    return best


def estimate_trajectory_sbl(range_seq, anchors, sigma_range, imu_deltas, sigma_imu,
                            depth_seq, sigma_depth, p_parent=None, loss="linear",
                            f_scale=1.345, return_cov=False):
    """SBL: 親機4トランスデューサへの距離 + IMU + 深度で子機軌道を推定する (MATH_SPEC §13)。

    各時刻に複数アンカー (既知配置) への距離があるので、単時刻でも多辺測量で3D位置が定まる
    (光学の方位は不要。方位グリッド探索も不要)。深度が鉛直 z を、IMU が時刻間を補強する。

    range_seq : (n,M) 各時刻・各アンカーへの距離 [m]
    anchors   : (M,3) トランスデューサの既知位置 [m] (親機座標)。親機が波で揺れる場合は
                (n,M,3) の時刻ごとに動くアンカー列も受ける (§13.5。推定姿勢で回した位置)。
    sigma_range: 各測距ノイズ [m]
    imu_deltas: (n-1,3) 世界座標の時刻間変位 [m]。None なら IMU 拘束なし。
    depth_seq : (n,) 深度 [m, 下が正]。None なら深度拘束なし。
    戻り値    : 推定軌道 X_hat (n,3)。truth は参照しない (MBD)。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    range_seq = np.asarray(range_seq, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    per_step = anchors.ndim == 3          # (n,M,3): 時刻ごとに動くアンカー (親機波動揺の補正)
    n, m = range_seq.shape
    sqrtW_r = 1.0 / sigma_range
    use_imu = imu_deltas is not None
    if use_imu:
        imu_deltas = np.asarray(imu_deltas, dtype=float)
        sqrtW_imu = 1.0 / np.broadcast_to(np.asarray(sigma_imu, dtype=float), (3,))
    use_depth = depth_seq is not None
    if use_depth:
        depth_seq = np.asarray(depth_seq, dtype=float).reshape(n)
        sqrtW_depth = 1.0 / sigma_depth

    # 初期値: 水平はアンカー配置の重心、鉛直は深度 (無ければ 0)
    center = anchors.reshape(-1, 3).mean(axis=0)
    x0 = np.tile(center, (n, 1))
    x0[:, 2] = -depth_seq if use_depth else -1.0

    def stacked(xflat):
        X = xflat.reshape(n, 3)
        parts = []
        for k in range(n):       # 各アンカーへの距離残差 (多辺測量)
            a_k = anchors[k] if per_step else anchors
            d_hat = np.linalg.norm(a_k - X[k], axis=1)
            parts.append(sqrtW_r * (range_seq[k] - d_hat))
        if use_imu:
            for k in range(n - 1):
                parts.append(sqrtW_imu * ((X[k + 1] - X[k]) - imu_deltas[k]))
        if use_depth:
            for k in range(n):
                parts.append(np.array([sqrtW_depth * depth_residual(X[k], depth_seq[k])]))
        return np.concatenate(parts)

    M = n * m + (3 * (n - 1) if use_imu else 0) + (n if use_depth else 0)

    def stacked_jac(xflat):      # 多辺測量の解析ヤコビアン (MATH_SPEC §13, §4.3)
        X = xflat.reshape(n, 3)
        J = np.zeros((M, 3 * n))
        r = 0
        for k in range(n):       # r_i=range_i-||a_i-x||: ∂/∂x = (a_i-x)/||a_i-x||
            a_k = anchors[k] if per_step else anchors
            diff = a_k - X[k]                      # (m,3)
            d_hat = np.linalg.norm(diff, axis=1)
            d_hat = np.where(d_hat < 1e-12, 1e-12, d_hat)
            J[r:r + m, 3 * k:3 * k + 3] = sqrtW_r * (diff / d_hat[:, None])
            r += m
        if use_imu:
            for k in range(n - 1):
                for a in range(3):
                    J[r + a, 3 * k + a] = -sqrtW_imu[a]
                    J[r + a, 3 * (k + 1) + a] = sqrtW_imu[a]
                r += 3
        if use_depth:
            for k in range(n):
                J[r, 3 * k + 2] = sqrtW_depth
                r += 1
        return J

    sol = _solve_least_squares(stacked, x0.ravel(), loss=loss, f_scale=f_scale,
                               jac=stacked_jac)
    X_hat = sol.x.reshape(n, 3)
    if return_cov:               # 各時刻 3x3 共分散 (SBL の GDOP, MATH_SPEC §4.5)
        Jw = stacked_jac(sol.x)
        cov_full = np.linalg.pinv(Jw.T @ Jw)
        cov = np.array([cov_full[3 * k:3 * k + 3, 3 * k:3 * k + 3] for k in range(n)])
        return X_hat, cov
    return X_hat


def optical_health_mask(detected, threshold=0.2, hysteresis=0.05, window=5):
    """光学リンクの健全性から、各時刻で方位/仰角を使うか (use-optical) を決める (MATH_SPEC §12)。

    直近 window フレームの**見失い率** (1 - 検出率) を見て、しきい値を超えたら光学を信頼せず
    フォールバックへ、下回ったら光学へ戻る状態機械 (ヒステリシスでチャタリング防止)。因果的
    (過去のみ参照) なのでオンライン運用に使える。

    detected  : (n,) bool。各時刻でビーコンを検出できたか。
    threshold : 見失い率がこれを超えると光学→フォールバックに切替。
    hysteresis: 戻る側のしきい値を threshold-hysteresis に下げる (チャタリング防止)。
    window    : 見失い率を測る移動窓のフレーム数。
    戻り値    : (n,) bool。True の時刻は光学 (角度) を使う。光学状態でも未検出フレームは False。
    """
    detected = np.asarray(detected, dtype=bool)
    n = len(detected)
    mask = np.zeros(n, dtype=bool)
    optical_state = True
    for k in range(n):
        lo = max(0, k - window + 1)
        dropout_rate = 1.0 - detected[lo:k + 1].mean()
        if optical_state and dropout_rate > threshold:
            optical_state = False
        elif (not optical_state) and dropout_rate < threshold - hysteresis:
            optical_state = True
        mask[k] = optical_state and detected[k]
    return mask


def estimate_trajectory_auto(z_seq, sigma_obs, detected, imu_deltas, sigma_imu,
                             depth_seq, sigma_depth, p_parent=None,
                             observe_from_parent=True, threshold=0.2,
                             hysteresis=0.05, window=5, n_azimuth_starts=12,
                             loss="huber", f_scale=1.345):
    """光学↔フォールバックを自動切替して軌道を推定する (MATH_SPEC §12)。

    光学が健全な時刻は (距離+方位+仰角)、見失い多発の時刻は (距離+深度+IMU) のみを使う
    1本のバッチ最小二乗。健全性は optical_health_mask が見失い率+ヒステリシスで判定する。

    初期値は**フォールバック解** (距離+IMU+深度の多スタート解, §11) を使う。これは方位込みで
    大域的に可観測なので、利用可能な光学フレームはそこから精緻化するだけでよい。

    z_seq    : (n,3) 観測 (d, theta, phi)。未検出時刻の角度は使われない (マスク)。
    detected : (n,) bool。各時刻でビーコンを検出できたか (切替判定に使う)。
    戻り値   : (X_hat (n,3), angle_mask (n,) bool)。mask は実際に角度を使った時刻。
    """
    if p_parent is None:
        p_parent = np.zeros(3)
    z_seq = np.asarray(z_seq, dtype=float)
    n = len(z_seq)
    mask = optical_health_mask(detected, threshold=threshold,
                               hysteresis=hysteresis, window=window)

    if mask.all():                     # 全時刻で光学健全 -> 逆変換初期化で十分 (多スタート不要)
        est = estimate_trajectory(
            z_seq, sigma_obs, imu_deltas=imu_deltas, sigma_imu=sigma_imu,
            p_parent=p_parent, observe_from_parent=observe_from_parent,
            loss=loss, f_scale=f_scale, z_depth_seq=depth_seq, sigma_depth=sigma_depth)
        return est, mask

    # 一部/全部の時刻で光学が使えない -> フォールバック解を大域初期値に (方位を解決, §11)
    x0 = estimate_trajectory_acoustic_inertial(
        z_seq[:, 0], sigma_obs[0], imu_deltas, sigma_imu, depth_seq, sigma_depth,
        p_parent=p_parent, observe_from_parent=observe_from_parent,
        n_azimuth_starts=n_azimuth_starts, loss=loss, f_scale=f_scale)

    if not mask.any():                 # 光学が全く使えない -> 純フォールバック
        return x0, mask

    est = estimate_trajectory(
        z_seq, sigma_obs, imu_deltas=imu_deltas, sigma_imu=sigma_imu,
        p_parent=p_parent, observe_from_parent=observe_from_parent, x0=x0,
        loss=loss, f_scale=f_scale, z_depth_seq=depth_seq, sigma_depth=sigma_depth,
        angle_mask=mask)
    return est, mask
