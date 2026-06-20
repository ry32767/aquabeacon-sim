# MATH_SPEC.md — 数式の正準定義

このファイルは AquaBeacon シミュレーションで使う数式の**唯一の正準定義**です。
実装 (`src/`) はこの定義に厳密に従ってください。式とコードが食い違う場合、このファイルが正です。

各数式は **3つの形式**で書いています。実装とレビューでは目的に応じて使い分けてください。

- **(A) 数式 (LaTeX)**: 人間がレビューするための表現。
- **(B) 擬似コード (Python)**: 実装の正準形。引数順・関数名・角度規約はこれに従う。
- **(C) 数値テストケース**: 実装が正しいか機械的に検証するための入出力例。`tests/test_math_cases.py` に落とす。

> **数式が崩れて読めない場合は、(B) 擬似コードと (C) 数値テストケースを正準として実装してよい。**
> LaTeX レンダリングに依存せず実装が一意に決まるよう、3形式を冗長に併記しています。

---

## 0. 座標系・記号・規約

### 0.1 座標系

- ワールド座標系 W: **親機(水上)を原点**とする ENU (East-North-Up)。
  - X 軸 = 東 (East)、Y 軸 = 北 (North)、Z 軸 = 上 (Up)。
  - 水面が Z ≈ 0。水中は Z < 0 (子機は親機より下にあるので z < 0)。
- 距離の単位は **メートル [m]**、角度の単位は内部計算ではすべて **ラジアン [rad]**。
  - 入出力で度を使う場合のみ境界で変換する。テストケースの角度は rad で書く。

### 0.2 記号対応表

| 記号 | 意味 | コード上の名前 |
|------|------|----------------|
| `p_P` | 親機の位置 (原点固定) | `p_parent` (既定 `[0,0,0]`) |
| `p_M = (x, y, z)` | 子機(機体)の位置。**推定対象** | `p_child` / `x_state` |
| `d` | 親機-子機間の距離 (音響が測る) | `dist` |
| `theta` | 方位角 azimuth | `az` |
| `phi` | 仰角 elevation (子機は下なので負) | `el` |
| `sigma_d`, `sigma_az`, `sigma_el` | 各観測の標準偏差 | `sigma_dist`, `sigma_az`, `sigma_el` |
| `W` | 重み行列 (= diag(1/sigma^2)) | `W` |
| `r` | 残差ベクトル | `residual` |
| `J` | ヤコビアン | `jac` |

### 0.3 観測の幾何 (反転可能にする)

提案構成では「**親機が子機を観測する**」(親機側のカメラが子機のライトの角度を測る) を既定とする。
ただし将来「子機のカメラが親機を観測する」構成にも対応できるよう、観測ベクトルの定義を1か所に集約する。

- 既定 (親機→子機): 相対ベクトル `v = p_M - p_P`。親機原点なら `v = p_M`。
- 反転 (子機→親機): `v = p_P - p_M`。`config.OBSERVE_FROM_PARENT = True/False` で切替。
- **以降の式はすべて `v = (vx, vy, vz)` を入力として書く。** 親機原点・既定構成では `v = (x, y, z)`。

---

## 1. 順変換: 位置 → 観測量 (sensors が使う)

真の相対ベクトル `v = (vx, vy, vz)` から、センサが測るはずの距離・角度を計算する。

### (A) 数式

```
d     = sqrt(vx^2 + vy^2 + vz^2)
theta = atan2(vy, vx)
phi   = atan2(vz, sqrt(vx^2 + vy^2))
```

### (B) 擬似コード

```python
import numpy as np

def forward_observation(v):
    """相対ベクトル v=(vx,vy,vz) [m] -> (d [m], theta [rad], phi [rad])"""
    vx, vy, vz = v
    d = np.sqrt(vx**2 + vy**2 + vz**2)
    theta = np.arctan2(vy, vx)               # azimuth, 範囲 (-pi, pi]
    phi = np.arctan2(vz, np.hypot(vx, vy))   # elevation, 子機が下なら負
    return np.array([d, theta, phi])
```

### (C) 数値テストケース

| 入力 v=(vx,vy,vz) [m] | 期待 d [m] | 期待 theta [rad] | 期待 phi [rad] | 許容 atol |
|----------------------|-----------|------------------|----------------|-----------|
| (3, 4, 0)            | 5.0       | 0.9272952180     | 0.0            | 1e-9 |
| (0, 0, -10)          | 10.0      | 0.0 *            | -1.5707963268  | 1e-9 |
| (1, 0, 0)            | 1.0       | 0.0              | 0.0            | 1e-9 |
| (0, 0, 0)            | 0.0       | 0.0 (規約)       | 0.0 (規約)     | 1e-9 |
| (6, 8, -7.5)         | 12.5      | 0.9272952180     | -0.6435011088  | 1e-9 |

> \* (0,0,-10) の theta は数学的に未定義 (vx=vy=0)。`atan2(0,0)=0` を採用する規約とする。
> ノイズフリーの推定では theta が未定義でも d と phi から z を復元できるので問題ない。

---

## 2. 逆変換: 観測量 → 位置 (estimator の閉形式・初期値に使う)

距離・角度から相対ベクトルを直接復元する。**観測が1組あれば 3次元位置は一意**。
これは球面座標→直交座標の変換そのもの。ノイズが無ければこれだけで真値に一致する。

### (A) 数式

```
vx = d * cos(phi) * cos(theta)
vy = d * cos(phi) * sin(theta)
vz = d * sin(phi)
```

### (B) 擬似コード

```python
def inverse_observation(d, theta, phi):
    """(d, theta, phi) -> 相対ベクトル v=(vx,vy,vz) [m]"""
    vx = d * np.cos(phi) * np.cos(theta)
    vy = d * np.cos(phi) * np.sin(theta)
    vz = d * np.sin(phi)
    return np.array([vx, vy, vz])
```

### (C) 数値テストケース (1の逆。往復で元に戻ること)

| 入力 (d, theta, phi)            | 期待 v=(vx,vy,vz) [m]      | 許容 atol |
|---------------------------------|---------------------------|-----------|
| (5.0, 0.9272952180, 0.0)        | (3.0, 4.0, 0.0)           | 1e-9 |
| (10.0, 0.0, -1.5707963268)      | (0.0, 0.0, -10.0)         | 1e-9 |
| (12.5, 0.9272952180, -0.6435011088) | (6.0, 8.0, -7.5)      | 1e-9 |

**往復テスト (test_roundtrip.py)**: 任意の `v` について
`inverse_observation(*forward_observation(v))` が元の `v` に戻る (atol=1e-9)。
ただし `v=(0,0,0)` と theta 未定義ケースは除外する。

---

## 3. 観測モデルと残差 (estimator の最小二乗で使う)

ノイズがある実観測では、逆変換の一発解ではなく、残差の二乗和を最小化する。

### 3.1 観測モデル h(x)

推定状態 `x = (x, y, z)` (= 子機位置) から、観測の予測値を返す。1節の forward と同じ。

```python
def h(x_state, p_parent=np.zeros(3), observe_from_parent=True):
    v = (x_state - p_parent) if observe_from_parent else (p_parent - x_state)
    return forward_observation(v)   # -> (d_hat, theta_hat, phi_hat)
```

### 3.2 残差 r(x)

実測 `z = (d, theta, phi)` と予測 `h(x)` の差。**角度成分は必ず正規化する。**

### (A) 数式

```
r_d     = d     - d_hat
r_theta = wrap(theta - theta_hat)
r_phi   = wrap(phi   - phi_hat)
wrap(a) = atan2(sin(a), cos(a))      # 角度差を (-pi, pi] に畳む
```

### (B) 擬似コード

```python
def wrap_angle(a):
    """角度差を (-pi, pi] に正規化。+-pi 境界の不連続を防ぐ。"""
    return np.arctan2(np.sin(a), np.cos(a))

def residual(x_state, z_meas, p_parent=np.zeros(3), observe_from_parent=True):
    d_hat, th_hat, ph_hat = h(x_state, p_parent, observe_from_parent)
    d, th, ph = z_meas
    return np.array([
        d - d_hat,
        wrap_angle(th - th_hat),
        wrap_angle(ph - ph_hat),
    ])
```

### (C) 数値テストケース

| x_state         | z_meas (d,theta,phi)             | 期待 residual           | atol |
|-----------------|----------------------------------|-------------------------|------|
| (3,4,0)         | (5.0, 0.9272952180, 0.0)         | (0, 0, 0)               | 1e-9 |
| (3,4,0)         | (5.1, 0.9272952180, 0.0)         | (0.1, 0, 0)             | 1e-9 |
| (1,0,0)         | (1.0, 3.1415926536, 0.0)         | (0, 3.1415926536, 0) ** | 1e-6 |

> \*\* wrap のテスト: theta_hat=0, theta=pi の差は pi。`pi - 0 = pi` で wrap して `pi` のまま。
> 一方 theta=−pi (= +pi と同一方向) なら `wrap(-pi - 0) = -pi`。境界の符号が安定することを確認する。

---

## 4. 重み付き最小二乗 (estimator 本体)

### 4.1 重み行列

各観測の標準偏差から重みを作る。精度の良い観測ほど重い。

### (A) 数式

```
W = diag(1/sigma_d^2, 1/sigma_theta^2, 1/sigma_phi^2)
```

注意: `sigma_d` は [m]、`sigma_theta`, `sigma_phi` は [rad]。
角度の sigma は (角度精度 0.3 deg) → `np.deg2rad(0.3)` で rad に直してから使う。

### (B) 擬似コード

```python
def weight_matrix(sigma_dist, sigma_az, sigma_el):
    return np.diag([1.0/sigma_dist**2, 1.0/sigma_az**2, 1.0/sigma_el**2])
```

### 4.2 目的関数と解法

### (A) 数式

```
x_hat = argmin_x  r(x)^T W r(x)
```

`h` が非線形なので反復解法を使う。ガウス・ニュートン更新:

```
x_{k+1} = x_k + (J^T W J)^{-1} J^T W r(x_k)
J = ∂h/∂x   (3x3 ヤコビアン)
```

### (B) 擬似コード (scipy 推奨。まず数値ヤコビアンで動かす)

```python
from scipy.optimize import least_squares

def estimate_position(z_meas, sigma, p_parent=np.zeros(3),
                      observe_from_parent=True, x0=None):
    """
    z_meas: (d, theta, phi) 観測
    sigma:  (sigma_dist, sigma_az[rad], sigma_el[rad])
    x0:     初期値。None なら 2節の逆変換で算出する。
    """
    sd, sa, se = sigma
    sqrtW = np.array([1.0/sd, 1.0/sa, 1.0/se])  # least_squares は残差を渡すので sqrt(W) を残差に掛ける

    if x0 is None:
        v0 = inverse_observation(*z_meas)
        x0 = (p_parent + v0) if observe_from_parent else (p_parent - v0)

    def weighted_residual(x_state):
        return sqrtW * residual(x_state, z_meas, p_parent, observe_from_parent)

    sol = least_squares(weighted_residual, x0, method='lm')
    return sol.x
```

> 単一観測 (d,theta,phi の3式) で未知数3 (x,y,z) なので、ちょうど決定系。
> ノイズフリーなら逆変換の x0 が既に厳密解で、least_squares は即収束する。
> **まず数値ヤコビアンで動かし**、速度・精度が問題になったら 4.3 の解析ヤコビアンを `jac=` に渡す。

### 4.3 解析ヤコビアン (任意。最適化のため。付録扱い)

`v = (vx,vy,vz)`, `d = ||v||`, `rho = sqrt(vx^2+vy^2)` として、観測 (d,theta,phi) の v に関する偏微分:

```
∂d/∂v     = (vx/d,  vy/d,  vz/d)
∂theta/∂v = (-vy/rho^2,  vx/rho^2,  0)
∂phi/∂v   = (-vx*vz/(d^2*rho),  -vy*vz/(d^2*rho),  rho/d^2)
```

`x_state` に対するヤコビアンは、既定構成 (observe_from_parent=True) では `∂v/∂x = I` なので上式そのまま。
反転構成では符号が反転する (`∂v/∂x = -I`)。`rho=0` (真上・真下) では theta/phi の微分が特異になるので、その近傍は数値ヤコビアンにフォールバックする。

### (C) 数値テストケース (estimator 全体)

1. **ノイズフリー一致**: `z = forward_observation((6,8,-7.5))` を入力に `estimate_position` を呼ぶと、
   結果が `(6,8,-7.5)` に一致する (atol=1e-6)。→ `test_noise_free.py`
2. **重みの効き**: sigma_dist を極端に大きく (距離を信用しない) すると、解が角度方向の制約を優先することを確認する (定性的テスト)。
3. **収束**: 初期値を真値から 5 m ずらしても、ノイズフリーなら真値に収束する (atol=1e-6)。

### 4.4 ロバスト推定 (外れ値対策, M推定)

§4.2 の純最小二乗 (L2) は、外れ値 (§8.3: ライト見失い・音響マルチパス) に弱い。1つの
大きな残差が二乗で効き、解全体を引っぱる。これを抑えるため、二乗の代わりにロバスト損失
`ρ(·)` を最小化する M推定を使う:

```
x_hat = argmin_x  Σ_i ρ( r_i(x) / σ_i )        (r_i / σ_i は §4.1 の sqrt(W) 正規化残差)
```

`ρ` の例 (`u = r/σ`, しきい値 `c = f_scale`):

```
linear (L2): ρ(u) = u^2                                  … 従来。外れ値に弱い
huber      : ρ(u) = u^2            (|u| <= c)
             ρ(u) = 2c|u| - c^2    (|u| >  c)            … c=1.345σ で内れ値ほぼL2, 外れ値は線形
cauchy     : ρ(u) = c^2 · ln(1 + (u/c)^2)                … 外れ値をさらに強く減衰
```

実装は `scipy.optimize.least_squares(loss=..., f_scale=c)` (method='trf')。`loss='linear'`
のときは従来通り LM で解き、結果は完全に一致する (後方互換)。`f_scale` は σ 単位の
内れ値しきい値で、既定 `1.345` (Huber の 95% 効率点)。設定は `config.toml [estimator]`。

**冗長性が前提**: 単時刻 (観測3・未知数3) は冗長性ゼロでどれが外れ値か決められないため、
ロバスト損失は実質効かない。効くのは **IMU 拘束つき軌道推定 (§5)**。時刻間の IMU 拘束と
他時刻の観測が冗長性を作り、外れ値の時刻だけを減衰できる。

### (C) 数値テストケース (`tests/test_robust.py`)

1. **後方互換**: `loss='linear'` の `estimate_position`/`estimate_trajectory` が従来 (LM) と一致 (atol=1e-9)。
2. **ノイズフリー不変**: ノイズ0なら `loss='huber'/'cauchy'` でも真値に収束 (atol=1e-6)。
3. **外れ値棄却**: 軌道の数時刻に大外れ値を注入したとき、IMU拘束つきで
   `loss='huber'`/`'cauchy'` の RMSE が `loss='linear'` より明確に小さい。

---

## 5. 複数時刻・IMU 拘束 (サクセス段階。ミニマムでは実装しない)

> **ミニマム段階ではこの節を実装しないこと。** ROADMAP のサクセス段階で着手する。
> ここでは構造だけ定義し、テストケースは段階到達時に追記する。

### 5.1 状態ベクトル (複数時刻)

```
X = (x_1, x_2, ..., x_n)        各 x_k = (x,y,z) at time t_k
```

### 5.2 残差 (2種類)

観測残差 (各時刻の音響・光学。3節と同じ):

```
r_obs_k = z_k - h(x_k)
```

IMU 拘束残差 (時刻間の変位を縛る):

```
r_imu_k = (x_{k+1} - x_k) - delta_p_imu_k
```

`delta_p_imu_k` は IMU 加速度の二重積分による予測変位 (pre-integration)。

### 5.3 目的関数

```
X_hat = argmin_X  sum_k ||r_obs_k||^2_{W_obs} + sum_k ||r_imu_k||^2_{W_imu}
```

これは先行研究 Wang et al. (2026) 式(2) と同じファクターグラフ/バンドル調整の構造。
実装は `scipy.optimize.least_squares` に全残差を連結したベクトルを渡す形で始める。

### (C) 数値テストケース (Stage 2 到達時に追記。`tests/test_trajectory.py`)

1. **ノイズフリー一致**: 軌道の各時刻を `forward_observation` した観測列を入力に
   `estimate_trajectory` を呼ぶと、推定軌道が真の軌道に一致する (atol=1e-6)。
   IMU 拘束 (真の変位 `delta_p = diff(traj)`) の有無いずれでも一致する。
2. **IMU 拘束の効果**: ノイズ込みの観測列に対し、IMU 拘束を加えた推定の軌道 RMSE が、
   IMU なしの推定より小さくなる (複数 seed の平均で評価)。
   実装の `sigma_imu` (config.SIGMA_IMU) が観測由来の点間誤差より小さいとき成立する。
3. **形状**: 戻り値は入力軌道と同じ (n,3) 形状。
> 観測残差 `r_obs_k` は 3節と同一。IMU 残差は `r_imu_k = (x_{k+1}-x_k) - delta_p_imu_k`。
> 全残差を `least_squares` に連結して渡す (各ブロックに sqrt(W) を掛ける)。

---

## 6. ジオメトリ評価 (サクセス段階)

推定軌道から作った点群・形状を、既知物体 (キューブ等) と比較する。

### 6.1 指標

```
点群距離 RMS = sqrt( mean_i( min_dist(p_est_i, surface_true)^2 ) )
寸法誤差     = L_hat - L_true                 [mm]
体積誤差率   = (V_hat - V_true) / V_true * 100  [%]
```

先行研究のスケールバー誤差・物体寸法比較と同じ思想 (Lo et al. 2024 Table 5)。

### (C) テストケース

- 既知の一辺 L=0.5 m のキューブを「真の点群」として生成し、それ自身を推定点群として入れたとき、
  寸法誤差 ≈ 0、体積誤差率 ≈ 0 になる (恒等チェック)。
- 既知のスケール係数 (例 1.02 倍) をかけた点群を入れると、寸法誤差がその係数を正しく反映する。

### 6.2 ステレオ観測モデル (子機2カメラ・ジオメトリ用)

> **構成の区別 (重要)**
> - **位置推定 (測位)**: 親機の **1台**のカメラ (角度) + 音響 (距離) → 子機位置を推定 (§1〜§5)。
> - **ジオメトリ作成 (マッピング)**: 子機に搭載した **2台**のカメラ (ステレオ) で対象表面点を
>   観測し、**三角測量**で3D点群を作る (本節)。音響距離は測位側で併用 (融合)。
>
> ジオメトリの点群は「親機カメラ+音響」では作らない。子機ステレオが作る。

子機はベースライン `B` [m] 離れた左右カメラ `c_L, c_R` を持つ。カメラ位置 (外部パラメータ) は
自機の測位と校正により**既知**とみなす。真の表面点 `P` を各カメラが角度で観測する。

#### (A) 順変換: 点 P → 2カメラの方位/仰角

各カメラについて、相対ベクトル `v = P - c_cam` から方位角・仰角を測る (§1 の角度部分と同じ)。

```
for cam in {L, R}:
    v = P - c_cam
    az_cam = atan2(v_y, v_x)
    el_cam = atan2(v_z, hypot(v_x, v_y))
観測 = (az_L, el_L, az_R, el_R)        # 距離は測らない。角度のみ x2
```

各カメラの観測に独立な正規ノイズ `N(0, sigma_cam)` を加える (`sigma_cam` は1カメラの角度精度)。

#### (B) 逆変換: 2方位 + カメラ位置 → 3D点 (三角測量・中点法)

各カメラの単位視線ベクトル (距離1の球面→直交):

```
u_cam = (cos(el)cos(az), cos(el)sin(az), sin(el))
```

2直線 `L1(s)=c_L + s·u_L`, `L2(t)=c_R + t·u_R` の最近接点の中点を P_hat とする:

```
w0 = c_L - c_R
b  = u_L · u_R
d  = u_L · w0 ;   e = u_R · w0
denom = 1 - b^2                 # 視線が平行なら 0 (特異 → 中点にフォールバック)
s = (b·e - d) / denom
t = (e - b·d) / denom
P_hat = 0.5 · ( (c_L + s·u_L) + (c_R + t·u_R) )
```

ノイズが無く2視線が交われば `P_hat = P` に厳密一致する。

#### 精度の効き方

ステレオの奥行き (depth Z) 誤差と横方向誤差は近似的に:

```
横方向誤差   ≈ Z · sigma_cam
奥行き誤差   ≈ Z^2 · sigma_cam / B
```

- **距離 Z が近いほど精度が上がる** (子機は対象に接近して撮る → 親機測位の d=12.5m より遥かに高精度)。
- **ベースライン B が長いほど奥行き精度が上がる**。
- 単眼 (親機カメラ) は奥行きを角度だけでは出せず音響に頼るが、ステレオは2視線の交点で出せる。

#### (C) 数値テストケース (`tests/test_stereo.py`)

1. **三角測量の恒等 (ノイズフリー復元)**: `c_L=(-B/2,0,0)`, `c_R=(B/2,0,0)` (例 B=0.1) で、
   いくつかの点 `P` について 順変換 (ノイズ0) → 三角測量 が `P` に一致する (atol=1e-9)。
2. **精度の傾向**: ノイズを加えたとき、ベースライン B を大きく / 距離 Z を小さくすると、
   復元点の誤差 (特に奥行き) が下がる (複数 seed の平均で評価)。
3. **キューブ復元**: 既知キューブ表面をステレオ復元した点群の寸法・体積誤差が妥当な範囲。

---

## 7. ノイズパラメータ初期値 (config.py の既定)

AquaBeacon の目標精度 (発表資料 参考資料②) を初期値とする。実機・LTspice 検証で得た値が出たら更新する。

| パラメータ | 記号 | 初期値 | 出典 |
|-----------|------|--------|------|
| 距離精度 | sigma_dist | 0.03 m (数cm) | 目標精度 距離精度 数cm / ~15m |
| 角度精度 | sigma_az, sigma_el | deg2rad(0.3) ≈ 0.00524 rad | 目標精度 角度 ±0.3° / ~10m |
| 相対位置(参考) | — | 水平 5–10 cm, 垂直 ~5 cm | 目標精度 相対位置精度 |
| GPS(親機絶対) | sigma_gps | 別途設定 (RTK相当なら数cm) | GPSの精度に依存 |
| 音響更新周期 | — | 5 Hz | 比較表 |
| 光学更新周期 | — | 30 Hz | 比較表 |

> 角度誤差は距離が遠いほど位置誤差に効く (距離 d でのライン誤差 ≈ d·sigma_angle)。
> 例: d=10 m, sigma=0.3° → 横方向誤差 ≈ 10 * 0.00524 ≈ 5.2 cm。感度解析でこの効きを確認すること。

---

## 8. 現実的センサ誤差モデル (実機前検証の高度化)

§1〜§7 は零平均ガウス誤差の理想モデルである。実機に近い検証のため、ここに追加の誤差源を
定義する。**§1〜§7 の式は変更しない**。本節は理想モデルへ重ねる「追加項」であり、
すべてのパラメータの既定値は『理想 (= §7 と完全一致)』になるよう選ぶ。
実装は `sensors.simulate_observation_realistic`、設定は `config.toml [error_model]/[acoustic]/[sync]`。

記号: 観測 `z = (d, θ, φ)`、真の相対距離 `d_true`、有効ノイズ `σ_eff`。

### 8.1 系統バイアス (calibration bias)

取付角・校正のズレによる定数オフセット。

```
z = z_ideal + b,   b = (b_d, b_θ, b_φ)
```

既定 `b = 0`。零平均ではないので、平均をとっても消えない (RMSE の下駄)。

### 8.2 距離依存ノイズ (range-dependent noise)

遠いほど SNR・分解能が落ちる効果を線形成長で近似する。

```
σ_d(d)   = σ_d0   · (1 + k_d · d)
σ_θ(d) = σ_φ(d) = σ_ang0 · (1 + k_a · d)
```

`k_d = dist_growth_per_m`, `k_a = range_growth_per_m`。既定 `k=0` で §7 の定数 σ に一致。

### 8.3 外れ値 (outliers)

ライト見失い・誤検出・音響マルチパスによる大きな誤差。各観測成分 `i` が独立に確率
`p = outlier_rate` で跳ねる混合モデル:

```
z_i += ε_i,   ε_i ~ N(0, (s · σ_eff,i)^2)   (確率 p),   ε_i = 0 (確率 1-p)
```

`s = outlier_scale` (既定 20)。既定 `p=0` で外れ値なし。ロバスト推定の必要性を示す入力。

### 8.4 音速ズレ (sound-speed mismatch)

音響測距は飛行時間 `τ = d_true / c_true` を測り、推定側は仮定音速 `c_assumed` で距離換算する:

```
d_meas = τ · c_assumed = d_true · (c_assumed / c_true)
```

`c_true ≠ c_assumed` なら**距離の系統スケール誤差**。海中音速は水温・塩分・水深で 1450〜1550 m/s。
既定 `c_true = c_assumed = 1500` で誤差なし。(水深依存の音速プロファイル SVP は将来拡張。)

### 8.5 時刻同期 (optical/acoustic latency)

光学 (30 Hz) と音響 (5 Hz) はサンプル時刻が異なる。角度は光学時刻 `t` の位置、距離は
音響時刻 `t - Δt` の位置を指す。子機速度 `v` のとき:

```
角度 (θ, φ): 位置 p(t) から
距離 d:      位置 p(t) - v·Δt から  (Δt = acoustic_latency_s)
```

既定 `Δt = 0` で同時刻。`v·Δt` のぶん距離と角度が不整合になり、融合精度を下げる。

### (C) 数値テストケース (`tests/test_sensor_realism.py`)

1. **理想一致**: 全既定で `simulate_observation_realistic(p, σ, seed)` ==
   `simulate_observation(p, σ, seed)` (atol=1e-12)。後方互換の保証。
2. **音速スケール (ノイズ0)**: `c_assumed/c_true = 1.02` → `d_meas = 1.02 · d_true` (rtol=1e-9)。
3. **バイアス**: `b_d = 0.5` → 多数平均の距離が `d_true + 0.5` に寄る (許容 σ/√N)。
4. **距離依存σ**: `effective_sigma(d, σ, k_a)` が `d` に対し線形増加。
5. **外れ値率**: `outlier_rate = 0.3`、多数試行で大誤差成分の割合 ≈ 0.3 (±許容)。
6. **時刻同期 (ノイズ0)**: `v, Δt ≠ 0` で距離が `‖p − vΔt‖` に一致 (rtol=1e-9)。

---

## 9. 親機光学リンク: 水中の減衰・拡散モデル

親機カメラによる角度追跡の現実性を上げる。光は水中で**吸収**と**散乱**を受けて減衰し、
遠い/濁るほど受光信号 (SNR) が落ちて角度精度が悪化し、ついには**ビーコンを見失う**
(ドロップアウト=外れ値)。§8.2 の線形 `range_growth` を**物理ベースに置き換える**追加項。
実装は `sensors.optical_*` と `simulate_observation_realistic(optical_model=...)`、設定は
`config.toml [optical]`。既定 `enable=false` では従来と一致。

記号: 距離 (光路長) `d` [m]、ビーム減衰係数 `c = a + b` [1/m] (吸収+散乱)、基準距離 `d_ref`。

### 9.1 減衰と受光信号

```
透過率   T(d) = exp(-c · d)                              (Beer–Lambert)
信号比   R(d) = (d_ref / d)^2 · exp(-c · (d - d_ref))    (幾何拡散 1/d^2 × 差分透過)
```

`R(d_ref) = 1`。`c` の目安: 清澄な外洋 ~0.05、沿岸 ~0.2–0.4、濁り ~0.5–2 [1/m]。

### 9.2 SNR と角度ノイズ

受光信号比から SNR を作り、重心推定の角度精度を SNR の逆数で悪化させる:

```
SNR(d)   = snr_ref · R(d)^p
σ_ang(d) = σ_floor + (σ_ref - σ_floor) / R(d)^p
```

- `p = snr_exponent`: 信号→SNR の依存 (1 = 後方散乱/背景律速、0.5 = ショットノイズ律速)。
- `σ_floor`: ベストケース角度ノイズ (画素量子化・校正限界)。`σ_ref`: `d_ref` での角度ノイズ。
- `d = d_ref` で `σ_ang = σ_ref`。近いほど `σ_floor` に漸近、遠い/濁るほど発散的に増大。
- この `σ_ang(d)` が観測の方位・仰角ノイズ σ_az, σ_el を置き換える (§7 の零平均ガウスの幅)。

### 9.3 ドロップアウト (ビーコン見失い)

SNR が検出しきい値 `snr_min` を下回ると、誤検出・追跡ロストで角度が大きく飛ぶ。確率:

```
p_drop(d) = dropout_max / (1 + exp( (4/snr_min) · (SNR(d) - snr_min) ))
```

`SNR >> snr_min` でほぼ 0、`SNR = snr_min` で `dropout_max/2`、`SNR → 0` で `dropout_max`。
ドロップ時は方位・仰角に一様乱数 `U(-Δ, Δ)` (`Δ = dropout_jump`) を加える = **外れ値**。
これは §4.4 のロバスト推定 (IMU拘束つき軌道) で抑えられる。深い水深・濁りで多発する。

### (C) 数値テストケース (`tests/test_optical.py`)

1. **基準点**: `R(d_ref) = 1`、`SNR(d_ref) = snr_ref`、`σ_ang(d_ref) = σ_ref` (rtol=1e-9)。
2. **単調性**: `d` 増 または `c` 増で `SNR` 減・`σ_ang` 増。`σ_ang ≥ σ_floor`。
3. **ドロップアウト**: `p_drop ∈ [0, dropout_max]`、深い/濁るほど増加、`SNR=snr_min` で `dropout_max/2`。
4. **後方互換**: `optical_model=None` で `simulate_observation_realistic` が §8 既定と不変。
5. **角度劣化**: 浅い (近) より深い (遠) で方位・仰角ノイズの実測ばらつきが増大 (統計)。

---

## 10. 深度センサ (圧力) による鉛直拘束

子機に**圧力センサ**を載せ、水面 (親機, z=0) 基準の絶対深度を直接測る。光学 (角度) ×
音響 (距離) に**第4の観測**を足して位置推定に融合する。深度は**距離・濁りに依存せず鉛直 z を
直接拘束**するので、光学が苦手な深い/濁った水 (§9) で特に効く。実装は `sensors.simulate_depth`
と `estimator.estimate_position(z_depth=...)` / `estimate_trajectory(z_depth_seq=...)`、設定は
`config.toml [depth]`。座標は Z=上なので深度 (下が正) は `-z`。

### 10.1 観測モデルと残差

```
深度観測   z_depth = -z_child + b_depth + N(0, σ_depth)     (b_depth: 海面気圧・潮位バイアス)
予測深度   h_depth(x) = -z                                   (x = (x,y,z))
深度残差   r_depth(x) = z_depth - h_depth(x) = z_depth + z
```

### 10.2 融合最小二乗

§4 の重み付き最小二乗に深度残差を重み `1/σ_depth^2` で連結する:

```
x_hat = argmin_x  ρ( r_obs(x) ⊘ σ_obs ) + ρ( r_depth(x) / σ_depth )
        (⊘ は成分ごとの σ 正規化、ρ は §4.4 の損失。観測 (d,θ,φ) と深度を同時に最小化)
```

軌道推定 (§5) では各時刻 k に深度残差 `r_depth,k = z_depth,k + z_k` を加える。

### 10.3 冗長性とロバスト性

観測 (d, θ, φ) のみだと**観測3・未知数3で冗長性ゼロ** → 単時刻ではロバスト損失 (§4.4) が
外れ値を棄却できない。**深度を足すと観測4・未知数3で冗長度1** となり、単時刻でも
ロバスト推定が外れ値 (特に距離 d・仰角 φ の異常) を識別・減衰できる。
ただし方位 θ は深度では拘束されない (θ は水平面内の向き) ので、θ の外れ値は救えない点に注意。

### (C) 数値テストケース (`tests/test_depth.py`)

1. **観測モデル**: `predicted_depth((x,y,z)) = -z`、`depth_residual` がノイズ0で真値で0。
2. **ノイズフリー一致**: 深度込みでもノイズ0なら `estimate_position` が真値に一致 (atol=1e-6)。
3. **後方互換**: `z_depth=None` で従来 (深度なし) と完全一致。
4. **z 精度向上**: 仰角ノイズが大きい条件で、深度ありの z 軸 RMSE が深度なしより小さい (統計)。
5. **単時刻ロバスト棄却**: 仰角 φ に外れ値を注入したとき、深度あり + `loss='huber'` の
   位置誤差が深度なし/L2 より小さい (冗長性が単時刻ロバストを可能にする)。
