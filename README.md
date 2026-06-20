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
python scripts/run_visualize.py     # 発表用の図・アニメ生成 (figures/ にシナリオ別フォルダで出力)
```

`run_mapping.py` / `run_spec.py` は数値結果を **`results/`** に JSON / CSV でも保存する
(後から比較・解析するため。`results/` は `.gitignore` 済み)。

### 設計スペックシート (`run_spec.py`)

感度掃引を「**目標精度を満たすための設計要求**」に逆算する MBD の成果物。
例: 「測位 RMSE ≤ 100 mm には 距離 d ≤ 15 m / 角度ノイズ ≤ 0.36°」
「マッピング寸法誤差 ≤ 30 mm には standoff ≤ 1.7 m / ベースライン ≥ 0.14 m /
σ_cam ≤ 0.08° / フレーム ≥ 30」。目標値と探索グリッドは `config.toml [spec]` で編集。
出力は表 (コンソール) + `figures/spec/design_spec.png` + `results/run_spec.{json,csv}`。

### ロバスト推定 (`config.toml [estimator]`)

外れ値 (ライト見失い・音響マルチパス, MATH_SPEC §8.3) に強い M推定 (MATH_SPEC §4.4)。
`estimate_position` / `estimate_trajectory` に `loss` 引数を追加 (`"linear"`(既定,純L2) /
`"huber"` / `"cauchy"` / `"soft_l1"`)。**既定 `linear` は従来と完全一致** (後方互換)。
ロバスト損失は redescending の悪い極小を避けるため、内部で **L2解からウォームスタート**する。
冗長性のある **IMU拘束つき軌道推定で効く** (単時刻は観測3・未知数3で冗長性ゼロのため効かない)。
`run_robust.py` のデモでは、外れ値下で **RMSE 411 → 30 mm (93%改善)**。
`config.toml [estimator] loss` で全スクリプト既定を切替可能。

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
