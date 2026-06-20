"""truth.py — ① 真値生成 (MATH_SPEC §0)。

真値を知ってよい層。Stage 1 では単一の子機真位置を返すだけ。
Stage 2 で軌道・既知物体の真形状を追加する。

注意: このモジュールを estimator/geometry から import してはならない (MBD 層分離)。
"""
import numpy as np

from src import config


def true_child_position(p_child=None):
    """子機の真位置 p_M=(x,y,z) [m] を返す (Stage 1 は単一点)。

    p_child=None なら config.TRUE_CHILD_POSITION を使う。
    引数で上書きできるようにして、感度解析等で別の真値を扱えるようにする。
    """
    if p_child is None:
        p_child = config.TRUE_CHILD_POSITION
    return np.asarray(p_child, dtype=float).copy()


def demo_trajectory(n_points=None, area=None, depth=None,
                    n_legs=None, origin=None, depth_ripple=None):
    """可視化デモ用の芝刈り (boustrophedon) 軌道を返す (n_points, 3) [m]。

    引数 None の項目は config (config.toml の [demo_trajectory]) を使う。

    Stage 2 のダブル芝刈りマッピングの先取りイメージ。乱数を使わず決定的。
    本格的な Stage 2 軌道は今後 truth.py に正式実装する (ここは発表デモ用の簡易版)。

    area=(W,H) [m]: XY 平面の掃引範囲。n_legs: 往復の本数。
    depth [m]: 基準深さ (負)。depth_ripple [m]: 3D で見やすくする微小な深さ変動。
    """
    n_points = config.DEMO_N_POINTS if n_points is None else n_points
    area = config.DEMO_AREA if area is None else area
    depth = config.DEMO_DEPTH if depth is None else depth
    n_legs = config.DEMO_N_LEGS if n_legs is None else n_legs
    origin = config.DEMO_ORIGIN if origin is None else origin
    depth_ripple = config.DEMO_DEPTH_RIPPLE if depth_ripple is None else depth_ripple
    W, H = area
    x0, y0 = origin
    y_legs = np.linspace(y0, y0 + H, n_legs)
    pts_per_leg = max(2, n_points // n_legs)
    xs, ys = [], []
    for i, yy in enumerate(y_legs):
        xline = np.linspace(x0, x0 + W, pts_per_leg)
        if i % 2 == 1:
            xline = xline[::-1]            # 折り返し (芝刈り)
        xs.extend(xline)
        ys.extend([yy] * pts_per_leg)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # 進行に沿った微小な深さ変動 (決定的)。真上付近を避け 3D を見やすくする。
    s = np.linspace(0.0, 2 * np.pi, len(xs))
    zs = depth + depth_ripple * np.sin(s)
    traj = np.column_stack([xs, ys, zs])
    return traj[:n_points]


# ----------------------------------------------------------------------------
# Stage 2: マッピング用の真値 (軌道 + 既知物体)
# ----------------------------------------------------------------------------
def _lawnmower(x0, y0, W, H, n_legs, pts_per_leg, sweep_axis):
    """単一の芝刈り掃引を返す (n_legs*pts_per_leg, 2)。

    sweep_axis='x': 各レグは X 方向に走り、Y 方向にステップする。
    sweep_axis='y': 各レグは Y 方向に走り、X 方向にステップする (直交掃引)。
    """
    pts = []
    if sweep_axis == 'x':
        for i, yy in enumerate(np.linspace(y0, y0 + H, n_legs)):
            line = np.linspace(x0, x0 + W, pts_per_leg)
            if i % 2 == 1:
                line = line[::-1]
            for xx in line:
                pts.append((xx, yy))
    else:  # 'y'
        for i, xx in enumerate(np.linspace(x0, x0 + W, n_legs)):
            line = np.linspace(y0, y0 + H, pts_per_leg)
            if i % 2 == 1:
                line = line[::-1]
            for yy in line:
                pts.append((xx, yy))
    return np.asarray(pts, dtype=float)


def double_lawnmower_trajectory(area=None, depth=None, n_legs=None,
                                pts_per_leg=None, origin=None):
    """ダブル芝刈り (直交2掃引) 軌道を返す (n, 3) [m]  (ROADMAP Stage 2)。

    1掃引目は X 方向レグ (Y にステップ)、2掃引目は Y 方向レグ (X にステップ) の
    直交クロスハッチ。面をまんべんなく覆うマッピング軌道。
    深さは一定 (depth)。親機原点に対し真上 (rho=0) を通らないよう origin をずらす。
    乱数は使わない (決定的)。引数 None の項目は config ([trajectory]) を使う。
    """
    area = config.TRAJ_AREA if area is None else area
    depth = config.TRAJ_DEPTH if depth is None else depth
    n_legs = config.TRAJ_N_LEGS if n_legs is None else n_legs
    pts_per_leg = config.TRAJ_PTS_PER_LEG if pts_per_leg is None else pts_per_leg
    origin = config.TRAJ_ORIGIN if origin is None else origin
    W, H = area
    x0, y0 = origin
    p1 = _lawnmower(x0, y0, W, H, n_legs, pts_per_leg, 'x')
    p2 = _lawnmower(x0, y0, W, H, n_legs, pts_per_leg, 'y')
    xy = np.vstack([p1, p2])
    z = np.full(len(xy), float(depth))
    return np.column_stack([xy[:, 0], xy[:, 1], z])


def true_cube_pointcloud(side=None, center=None, n_per_edge=None):
    """既知キューブ (軸平行) の表面点群を返す (M, 3) [m]  (MATH_SPEC §6 テスト用)。

    一辺 side, 中心 center の立方体の6面に格子状に点を配置する。決定的 (乱数なし)。
    軸平行なので AABB の各辺長 = side に厳密一致する (恒等チェックに使える)。
    引数 None の項目は config ([cube]) を使う。
    """
    if side is None:
        side = config.CUBE_SIDE
    if center is None:
        center = config.CUBE_CENTER
    if n_per_edge is None:
        n_per_edge = config.CUBE_N_PER_EDGE
    center = np.asarray(center, dtype=float)
    h = side / 2.0
    lin = np.linspace(-h, h, n_per_edge)
    a, b = np.meshgrid(lin, lin)
    a = a.ravel()
    b = b.ravel()
    ones = np.full_like(a, h)
    faces = [
        np.column_stack([a, b,  ones]),   # z = +h
        np.column_stack([a, b, -ones]),   # z = -h
        np.column_stack([a,  ones, b]),   # y = +h
        np.column_stack([a, -ones, b]),   # y = -h
        np.column_stack([ ones, a, b]),   # x = +h
        np.column_stack([-ones, a, b]),   # x = -h
    ]
    pts = np.unique(np.vstack(faces), axis=0)   # 稜線の重複を除去
    return pts + center
