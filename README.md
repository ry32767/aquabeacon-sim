# AquaBeacon Simulation (MBD)

光学追跡(角度)×音響測距(距離)による水中位置推定と、その結果からのジオメトリ生成を、
**実機実験の前に**シミュレーションで検証するための MBD (モデルベース開発) 環境。

実装は Claude Code に委譲する。Claude Code はまず `CLAUDE.md` と `docs/MATH_SPEC.md` を読むこと。

## ドキュメントの読む順

1. `CLAUDE.md` — 作業方針・制約・自律修正ループ・完了条件
2. `docs/MATH_SPEC.md` — **数式の正準定義** (座標系・変換・最小二乗)。各式に LaTeX + 擬似コード + 数値テストケースを併記。
3. `docs/ROADMAP.md` — 実装段階 (Stage 0 足場 → 1 ミニマム → 2 サクセス)
4. `docs/VISUALIZATION.md` — 発表/デモ用の図・アニメ仕様 (`scripts/run_visualize.py`)

## セットアップ

```bash
pip install -r requirements.txt
```

## パラメータの調整 (ここを編集)

シミュレーションのパラメータは、リポジトリ直下の **`config.toml`** に集約しています。
ノイズ・座標・軌道・キューブ・モンテカルロ試行数・感度の振り幅・可視化の規模などを、
**コードを触らずテキストで一括変更**できます。編集後はそのまま各スクリプトを実行すれば反映されます。

```toml
[noise]
sigma_dist   = 0.03   # 距離 [m]
sigma_az_deg = 0.3    # 方位角 [deg]
[trajectory]
n_legs      = 4
```

- 行やセクションを消すと、`src/config.py` のデフォルト値が使われます (壊れません)。
- `config.toml` の読込には **Python 3.11+** が必要 (標準ライブラリ `tomllib`)。
  3.10 では `config.toml` は無視され、`src/config.py` のデフォルトで動きます。
- BOM 付き UTF-8 で保存されても読めるようにしてあります (Windows エディタ対策)。

## コマンド

```bash
pytest tests/                       # 全テスト
pytest tests/test_math_cases.py -v  # 数式の数値検証(最優先)
python scripts/run_minimum.py       # ミニマム: 単時刻の位置推定+RMSE
python scripts/run_mapping.py       # サクセス: 複数時刻+IMUの軌道推定 / キューブ寸法・体積
python scripts/run_sensitivity.py   # 感度解析(ノイズ・距離・角度を振る)
python scripts/run_spec.py          # 設計スペックシート: 目標精度→設計要求を逆算
python scripts/run_robust.py        # ロバスト推定デモ: 外れ値下で L2 vs Huber/Cauchy
python scripts/run_deepwater.py     # 深い水深(10-20m)テスト: 光減衰→精度劣化・見失い
python scripts/run_depth.py         # 深度センサ融合デモ: z軸精度↑・単時刻ロバスト
python scripts/run_no_optical.py    # 光学なしフォールバック: 距離+IMU+深度のみで測位
python scripts/run_opmap.py         # 2次元運用スペック: 濁り×水深の運用可能領域マップ
python scripts/run_switch.py        # 光学↔フォールバック自動切替: プルーム通過で切替維持
python scripts/run_visualize.py     # 発表用の図・アニメ生成 (figures/ にシナリオ別フォルダで出力)
```

### 光学↔フォールバック 自動切替 (`config.toml [switch]`)

光学リンクの健全性に応じて時刻ごとに光学とフォールバックを切り替える (MATH_SPEC §12)。
`estimator.estimate_trajectory_auto` は各時刻の検出フラグから**見失い率** (移動窓) を測り、
`dropout_threshold` を超えたらフォールバック、下回ったら光学へ戻す (ヒステリシスでチャタリング防止)。
切替は estimate_trajectory の**時刻別 angle_mask** に落ち、光学が使える区間は (d,θ,φ)、見失い区間は
距離+IMU+深度 (§11) の1本のバッチ最小二乗になる。`run_switch.py` (濁りプルーム通過で一時ブラックアウト)
では **素朴な光学維持 200mm / 常時フォールバック 58mm / 自動切替 33mm** と両モードのいいとこ取り。

### 2次元運用スペック (`run_opmap.py`)

水深 × 濁り c の格子で、各条件をどのモードで運用できるかを3色で塗り分ける
(`figures/opmap/operational_map.png`): **緑=光学** (見失い率 ≤ しきい値 かつ目標精度)、
**金=フォールバック** (光学不可だが距離+IMU+深度 §11 で目標達成 → 自動切替で継続)、
**赤=運用不可**。光学領域は濁り・深さで縮み、フォールバックは光を使わず濁り非依存で広く覆う。
破線は切替境界 (見失い率 = `config.toml [switch] dropout_threshold`)。この地図が自動切替の判断面。

`run_mapping.py` / `run_spec.py` は数値結果を **`results/`** に JSON / CSV でも保存する
(後から比較・解析するため。`results/` は `.gitignore` 済み)。

### 設計スペックシート (`run_spec.py`)

感度掃引を「**目標精度を満たすための設計要求**」に逆算する MBD の成果物。
例: 「測位 RMSE ≤ 100 mm には 距離 d ≤ 15 m / 角度ノイズ ≤ 0.36°」
「マッピング寸法誤差 ≤ 30 mm には standoff ≤ 1.7 m / ベースライン ≥ 0.14 m /
σ_cam ≤ 0.08° / フレーム ≥ 30」。目標値と探索グリッドは `config.toml [spec]` で編集。
出力は表 (コンソール) + `figures/spec/design_spec.png` + `results/run_spec.{json,csv}`。

さらに **運用可能な最大水深** (光学減衰 §9 + 深度センサ §10 の統合) を濁りごとに出す。
深度センサありで「ミッション精度 (既定 300 mm) を満たす最大水深」がどこまで伸びるかを
深度センサなしと比較する。例 (既定設定): clear water で **16.5 m → 19.5 m** に延伸、
turbid では延伸せず (光ビーコンの**検出限界**が先に縛るため深度センサでは救えない)。
深い水では精度律速 → 深度センサで延伸 → 最後は検出律速、という構造が読める。
ミッション精度は `config.toml [spec] op_depth_target_mm`。図は `figures/spec/operational_depth.png`。

### 光学なしフォールバック (距離+IMU+深度のみ)

光学追跡が使えない/失われた場合 (濁り水でビーコン見失い = 検出律速) のフォールバック
(MATH_SPEC §11)。`estimator.estimate_trajectory_acoustic_inertial` は**音響距離・IMU・深度**
のみで軌道を推定する。単時刻は方位が不可観測 (距離+深度の2拘束) だが、IMU が時刻間を繋ぐと
軌道が可観測になる (方位の局所解は開始方位のグリッド探索で回避)。
**光を使わないので水の濁り・深さによる光学劣化 (§9) に不感** — `run_no_optical.py` では、
光学の検出限界 (turbid 11.5m / coastal 14m / clear 20m) より深い水でも z を深度で締めつつ
測位を継続 (深さ17mで RMSE ~50mm、z ~20mm)。光学と相補の安全網。

### 深度センサ融合 (`config.toml [depth]`)

子機の圧力センサで**絶対深度 (=-z) を直接測り**、光学×音響に第4の観測として融合する
(MATH_SPEC §10)。`estimate_position(z_depth=..., sigma_depth=...)` /
`estimate_trajectory(z_depth_seq=...)`。既定 `z_depth=None` で従来と完全一致。効果は2つ:
- **鉛直 z を距離・濁りに依存せず拘束** → 光学が劣化する深い/濁った水で z 精度が激増
  (`run_depth.py`: 水深20mで z RMSE 2509→316mm、深さ15mで z 184→46mm)。
- **観測4 > 未知数3 の冗長性** → **単時刻でもロバスト推定が外れ値を棄却**できる
  (これまで単時刻は冗長性ゼロで不可だった。例: 仰角外れ値で L2 2343 → huber+深度 114mm)。

`config.toml [depth] sigma_m`(圧力センサ精度) で調整。

### ロバスト推定 (`config.toml [estimator]`)

外れ値 (ライト見失い・音響マルチパス, MATH_SPEC §8.3) に強い M推定 (MATH_SPEC §4.4)。
`estimate_position` / `estimate_trajectory` に `loss` 引数を追加 (`"linear"`(既定,純L2) /
`"huber"` / `"cauchy"` / `"soft_l1"`)。**既定 `linear` は従来と完全一致** (後方互換)。
ロバスト損失は redescending の悪い極小を避けるため、内部で **L2解からウォームスタート**する。
冗長性のある **IMU拘束つき軌道推定で効く** (単時刻は観測3・未知数3で冗長性ゼロのため効かない)。
`run_robust.py` のデモでは、外れ値下で **RMSE 411 → 30 mm (93%改善)**。
`config.toml [estimator] loss` で全スクリプト既定を切替可能。

### 親機光学リンク: 水中の減衰・拡散 (`config.toml [optical]`) と深水深テスト

光は水中で吸収・散乱して減衰する (MATH_SPEC §9)。遠い/濁るほど受光 SNR が落ち、
角度ノイズ σ_ang が増え、ついには**ビーコン見失い (ドロップアウト=外れ値)** が起きる。
モデルは透過率 `T=exp(-c·d)`・信号比 `R=(d_ref/d)²·exp(-c(d-d_ref))`・`σ_ang=σ_floor+(σ_ref-σ_floor)/R^p`。
`c` (ビーム減衰係数 [1/m]) が水の濁り (clear 0.05 / coastal 0.3 / turbid 1.0)。
既定 `[optical] enable=false` で従来と一致。`sensors.simulate_observation_realistic(optical_model=...)`。

`run_deepwater.py` は水深 5〜20m × 濁り clear/coastal/turbid で測位精度・見失いを掃引し、
深い濁った水での軌道を **linear vs robust** で比較する (見失いが多い領域でロバストが軌道を救う)。
深さ・濁りは `config.toml [deepwater]` で編集。出力は `figures/deepwater/` + `results/run_deepwater.{json,csv}`。

### 現実的センサ誤差モデル (`config.toml [error_model]`)

実機に近い検証用に、理想 (零平均ガウス) へ重ねる誤差源を用意 (MATH_SPEC §8):
系統バイアス・距離依存ノイズ・外れ値・**音速ズレ**・**光学/音響の時刻同期**。
既定はすべて『理想』で従来と完全一致。`[error_model] enable = true` で有効化すると、
`run_spec.py` の測位スペックがこの誤差込みで厳しくなる (例: バイアス5cm+音速ズレで 90→144 mm)。

## MBD の構造

| 層 | モジュール | 真値を見てよいか |
|----|-----------|------------------|
| ① 真値生成 | `src/truth.py` | ○ |
| ② センサモデル | `src/sensors.py` | ○ (真値→ノイズ付き観測) |
| ③ 位置推定 | `src/estimator.py` | **✗ 観測値のみ** |
| ④ ジオメトリ生成 | `src/geometry.py` | **✗** |
| ⑤ 評価 | `src/evaluation.py` | ○ (ここでだけ真値vs推定) |

`tests/test_separation.py` がこの分離を静的に強制する。

### カメラ構成 (2系統)

- **位置推定 (測位)**: 親機の **1台**のカメラ (角度) + 音響 (距離) → 子機位置・軌道を推定。
- **ジオメトリ作成 (マッピング)**: 子機の **2台**のカメラ (ステレオ) → 三角測量で3D点群 (MATH_SPEC §6.2)。

ジオメトリの点群は親機カメラではなく子機ステレオが作る。`config.toml` の `[stereo]` で
ベースライン・観測距離・カメラ角度ノイズを調整できる。

## 公開について

将来オープンソース化予定。座標系・記号・ノイズパラメータの出典は `docs/MATH_SPEC.md` に明記。
