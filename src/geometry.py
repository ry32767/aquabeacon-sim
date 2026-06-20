"""geometry.py — ④ ジオメトリ生成 (MATH_SPEC §6)。

真値を見てはならない層。推定軌道や推定点群を入力に、寸法・体積などの形状量を
計算する。真値との比較 (寸法誤差・体積誤差率) は evaluation 側で行う。

このモジュールは truth を import しない (MBD 層分離。test_separation が強制)。

- trajectory_to_pointcloud : 推定軌道 -> 点群 (現状はダウンサンプル付きの受け渡し)
- aabb_dimensions          : 軸平行バウンディングボックスの各辺長 (Lx,Ly,Lz)
- aabb_volume              : AABB の体積
- convex_hull_volume       : 凸包の体積 (掃引形状向け)
- cube_side_estimate       : キューブ一辺の推定値 L_hat (AABB 3辺の平均)
"""
import numpy as np


def _unit_from_bearing(az, el):
    """方位角・仰角 -> 単位視線ベクトル (距離1の球面→直交)。"""
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)])


def stereo_triangulate(bearings, cam_L, cam_R):
    """ステレオ2方位 + カメラ位置 -> 3D点 P_hat (中点法)  (MATH_SPEC §6.2)。

    bearings: (az_L, el_L, az_R, el_R) [rad]
    cam_L, cam_R: 左右カメラ位置 [m] (既知の外部パラメータ)
    戻り値  : 三角測量で復元した点 (3,)

    2視線の最近接点の中点を返す。視線が平行 (特異) ならカメラ中点にフォールバック。
    入力は観測 (方位) と既知カメラ位置のみ。真値は参照しない (MBD)。
    """
    az_L, el_L, az_R, el_R = bearings
    u_L = _unit_from_bearing(az_L, el_L)
    u_R = _unit_from_bearing(az_R, el_R)
    c_L = np.asarray(cam_L, dtype=float)
    c_R = np.asarray(cam_R, dtype=float)

    w0 = c_L - c_R
    b = float(u_L @ u_R)
    d = float(u_L @ w0)
    e = float(u_R @ w0)
    denom = 1.0 - b * b
    if abs(denom) < 1e-12:                 # 視線が平行 -> 中点にフォールバック
        return 0.5 * (c_L + c_R)
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    p_L = c_L + s * u_L
    p_R = c_R + t * u_R
    return 0.5 * (p_L + p_R)


def trajectory_to_pointcloud(trajectory, every=1):
    """推定軌道 (n,3) を点群に変換する (MATH_SPEC §6)。

    現状は軌道点そのものを点群として扱う (every>1 で間引き)。将来、各時刻に
    観測したサーフェス点を展開する拡張余地を残す受け渡し関数。
    """
    pts = np.asarray(trajectory, dtype=float).reshape(-1, 3)
    return pts[::every]


def aabb_dimensions(points):
    """点群の軸平行バウンディングボックスの各辺長 (Lx,Ly,Lz) [m] を返す。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts.max(axis=0) - pts.min(axis=0)


def aabb_volume(points):
    """AABB の体積 [m^3] = Lx*Ly*Lz。"""
    return float(np.prod(aabb_dimensions(points)))


def convex_hull_volume(points):
    """点群の凸包の体積 [m^3]。掃引形状の体積推定に使う。

    scipy.spatial.ConvexHull を使用。点が同一平面など退化する場合は 0 を返す。
    """
    from scipy.spatial import ConvexHull, QhullError
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:
        return 0.0


def cube_side_estimate(points):
    """キューブの一辺 L_hat [m] を推定する (AABB 3辺長の平均)。"""
    return float(np.mean(aabb_dimensions(points)))


def robust_dimensions(points, lo=2.0, hi=98.0):
    """外れ値にロバストな各辺長 (パーセンタイル幅) [m] を返す。

    AABB (max-min) は最外点1個に敏感で、ノイズで寸法が上振れする。
    各軸の [lo, hi] パーセンタイル幅を使うことで、ノイズ点の影響を抑える。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.percentile(pts, hi, axis=0) - np.percentile(pts, lo, axis=0)


def robust_volume(points, lo=2.0, hi=98.0):
    """ロバスト寸法に基づく体積 [m^3] = 積(ロバスト各辺長)。"""
    return float(np.prod(robust_dimensions(points, lo, hi)))


def robust_cube_side_estimate(points, lo=2.0, hi=98.0):
    """ロバスト寸法に基づくキューブ一辺の推定値 L_hat [m] (3辺長の平均)。"""
    return float(np.mean(robust_dimensions(points, lo, hi)))
