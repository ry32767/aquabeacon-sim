# ROADMAP.md — 実装段階

発表資料のクライテリア (ミニマム / サクセス / フルサクセス) に対応した実装段階。
Claude Code は、指示された段階まで実装する。**段階を飛ばさない。**

---

## Stage 0: 足場 (最初にやる)

- `src/config.py`: 座標系・ノイズ・幾何パラメータを一元管理 (MATH_SPEC §0, §7)。
- `src/sensors.py`: `forward_observation`, `inverse_observation` (MATH_SPEC §1, §2)。
- `tests/test_math_cases.py` を全部緑にする。
- 完了条件: `pytest tests/test_math_cases.py` が全 pass。

## Stage 1: ミニマム — 単時刻の位置推定

> 発表資料「ミニマム: 位置推定のみをプールで実証・評価」に対応。

- `src/truth.py`: 単一の子機真位置 `p_M` を返す (固定 or 軌道の1点)。
- `src/sensors.py`: 真値 + seed から (d, theta, phi) のノイズ付き観測を生成。
- `src/estimator.py`: `residual`, `wrap_angle`, `weight_matrix`, `estimate_position` (MATH_SPEC §3, §4)。
- `src/evaluation.py`: 真値 vs 推定の RMSE (X/Y/Z 別と合成)。
- `scripts/run_minimum.py`: 1点を推定 → RMSE を表示。
- 完了条件:
  - `test_noise_free`: ノイズ0で推定が真値に一致 (atol=1e-6)。
  - `test_separation`: estimator/geometry が truth を import していない。
  - `scripts/run_minimum.py` が動き、ノイズフリーで RMSE ≈ 0。
  - ノイズ込みの RMSE が物理的に妥当 (角度誤差×距離のオーダー。MATH_SPEC §7 参照)。

## Stage 2: サクセス — マッピング (ジオメトリ生成)

> 発表資料「サクセス: マッピングをプール/実海域で実証・評価」に対応。

- `src/truth.py`: ダブル芝刈り軌道 + 既知物体 (キューブ) の真の形状。
- 複数時刻の観測列を生成。
- `src/estimator.py`: 複数時刻・IMU 拘束を加えた最小二乗 (MATH_SPEC §5)。
- `src/geometry.py`: 推定軌道 → 点群 → 寸法・面積・体積 (MATH_SPEC §6)。`open3d` を使ってよい。
- `src/evaluation.py`: 寸法誤差・体積誤差率を追加。
- 完了条件:
  - MATH_SPEC §6 のジオメトリテストケースが pass。
  - 既知キューブの寸法・体積誤差が妥当な範囲。

## Stage 3: フルサクセス (今回は範囲外。将来)

- 水深 10–20 m を想定したパラメータでの感度解析。
- リアルタイム動作は今回行わない (発表資料の判断通り)。

---

## 感度解析 (Stage 1 完了後いつでも)

`scripts/run_sensitivity.py`:
- ノイズ (sigma_dist, sigma_angle) を振って RMSE がどう変わるか。
- 親機-子機距離 d を振って、角度誤差の効き (≈ d·sigma_angle) を確認。
- 仰角 phi を振って、真下付近 (rho≈0) で精度が落ちないか確認。
- これが実機実験の設計 (現実的な深度・距離) の根拠になる。
