# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業するときの前提・制約・確認手順・完了条件を定義します。
Claude Code は作業前に必ずこのファイルと `docs/MATH_SPEC.md` を読み、ここに書かれた方針に従ってください。

---

## Project Overview

- Project name: AquaBeacon Simulation (MBD)
- Purpose: 光学追跡(角度)×音響測距(距離)による水中位置推定と、その結果を使ったジオメトリ生成を、**実機実験の前に**シミュレーションで検証する。
- 開発思想: MBD (モデルベース開発)。**真値を持つ仮想世界**と**推定する側**を完全に分離する。
- Target users: 開発者本人(高校生)。将来的にオープンソース化。
- Language: Python 3.10+
- 主要ライブラリ: `numpy`, `scipy`, `matplotlib`, (ジオメトリ段階で) `open3d`
- 本リポジトリは Web アプリではない。ローカルで動く計算・解析スクリプト群である。

### MBD の鉄則 (最重要・絶対に破らないこと)

1. **真値生成 (truth) とセンサモデル (sensor) は真値を知ってよい。**
2. **推定アルゴリズム (estimator) とジオメトリ生成 (geometry) は、真値を絶対に参照してはならない。** 入力はセンサ値(ノイズ込)のみ。
3. **評価 (evaluation) のときだけ、真値と推定値を突き合わせる。**
4. もしこの分離を壊すコード(estimator が truth を import する等)を見つけたら、それはバグである。修正すること。

---

## First Steps

作業を始める前に、以下を順に確認してください。

1. `CLAUDE.md` (このファイル) — 全体方針
2. `docs/MATH_SPEC.md` — **数式の正準定義**。座標系・記号・変換式・最小二乗の組み方。実装はこの定義に厳密に従う。
3. `docs/ROADMAP.md` — 実装の段階 (ミニマム → サクセス)。どこまで作るか。
4. `README.md` — セットアップとコマンド
5. `tests/` の既存テスト — 何が検証されているか
6. `src/` の既存モジュール — 既存の設計と命名

実装に入る前に、必要なら以下を短く整理してください。

- 現在どのモジュールが存在し、どこまで実装済みか
- 実行可能なコマンド (`README.md` と実際のファイルで確認)
- 不明点 (あれば質問)

---

## Core Rules

- `docs/MATH_SPEC.md` の定義 (座標系・記号・式) を唯一の正とする。式を勝手に変えない。変更が必要と判断したら、実装前に理由を述べて質問する。
- 既存の設計・命名・ディレクトリ構成を尊重する。
- 依頼された範囲に集中し、関係ない変更を避ける。
- MBD の層分離 (truth / sensor / estimator / geometry / evaluation) を常に守る。
- 依存を追加するときは目的を明記する。標準は `numpy`, `scipy`, `matplotlib`, `open3d` のみ。
- 乱数を使うコードは必ず `seed` を引数で受け取り、再現可能にする。
- 軽微な判断は自律的に行ってよい。完了条件に大きく影響する不明点は質問する。
- 最後に、変更内容・検証結果・残課題を報告する。

---

## Repository Structure

```
aquabeacon-sim/
├── CLAUDE.md              # このファイル(方針)
├── README.md              # セットアップ・コマンド
├── config.toml            # パラメータ一括管理(ユーザが編集。src/config.py が読み込む)
├── requirements.txt       # 依存
├── docs/
│   ├── MATH_SPEC.md       # 数式の正準定義(実装はこれに従う)
│   └── ROADMAP.md         # 実装段階
├── src/
│   ├── config.py          # パラメータ一元管理(座標系・ノイズ・幾何)
│   ├── truth.py           # ① 真値生成(真値を知ってよい)
│   ├── sensors.py         # ② センサモデル 真値→ノイズ付き観測(真値を知ってよい)
│   ├── estimator.py       # ③ 位置推定(真値を見ない。入力は観測のみ)
│   ├── geometry.py        # ④ ジオメトリ生成(真値を見ない)
│   └── evaluation.py      # ⑤ 評価(ここでだけ真値vs推定)
├── tests/
│   ├── test_math_cases.py # MATH_SPECの数値テストケース(最優先で通す)
│   ├── test_roundtrip.py  # 順変換→逆変換で元に戻るか
│   ├── test_noise_free.py # ノイズ0なら推定が真値に一致するか
│   └── test_separation.py # MBD層分離が守られているか(estimatorがtruthを参照しない)
└── scripts/
    ├── run_minimum.py     # ミニマム: 単時刻の位置推定を1本通す
    └── run_sensitivity.py # 感度解析: ノイズ・距離・角度を振ってRMSEを出す
```

存在しないディレクトリは、作成するか、この記述を実態に合わせて修正してください。

---

## Commands

`README.md` と実ファイルを確認し、存在するものだけ実行してください。

```bash
# 依存インストール
pip install -r requirements.txt

# テスト(最優先)
pytest tests/ -v

# 数式の数値テストケースだけ実行
pytest tests/test_math_cases.py -v

# ミニマム構成を1本通す
python scripts/run_minimum.py

# 感度解析
python scripts/run_sensitivity.py
```

---

## Self-Verification Loop (自律修正の手順)

このプロジェクトでは、**テストケースが仕様の一部**です。Claude Code は次のループを自律的に回してください。

1. `docs/MATH_SPEC.md` の「数値テストケース」節を読む。
2. それを `tests/test_math_cases.py` のアサーションとして実装する(まだ無ければ作る)。
3. `pytest tests/test_math_cases.py -v` を実行する。
4. **失敗したら、実装(src/)を修正する。MATH_SPEC は変更しない。**
   - ただし、MATH_SPEC 自体に矛盾や誤りを発見した場合は、修正せず停止して報告する。
5. 全テストが通るまで 3〜4 を繰り返す。
6. 次に `tests/test_roundtrip.py`, `tests/test_noise_free.py`, `tests/test_separation.py` を同様に通す。
7. すべて緑になったら、`scripts/run_minimum.py` を実行し、出力(RMSE)が ROADMAP のミニマム基準を満たすか確認する。

**許容誤差の扱い**: 数値比較は厳密一致ではなく、MATH_SPEC が指定する許容誤差 (`atol`, `rtol`) を使う。指定が無い場合は浮動小数は `atol=1e-9` を既定とする。

**無限ループ防止**: 同じテストの修正を5回試して通らない場合は、ループを止め、何が起きているか(エラー内容・試したこと・推定原因)を報告して人間の判断を仰ぐ。

---

## When to Ask Questions

実装前に質問すべき場合:

- `docs/MATH_SPEC.md` の式と、実装で必要な情報に食い違いがある場合
- MATH_SPEC に矛盾・誤りを見つけた場合 (勝手に直さない)
- 角度の符号規約・座標軸の向きが、コード上のある場面で一意に定まらない場合
- ROADMAP のどの段階まで実装するか曖昧な場合
- 新しい依存ライブラリを追加する必要がある場合

---

## When You May Decide Autonomously

依頼範囲内なら自律判断してよい:

- 明らかなバグ修正、型・lint エラーの修正
- MATH_SPEC の数値テストケースを通すための src 実装の修正
- docstring・コメントの追加、README の軽微な補足
- テストの追加 (既存の検証を壊さない範囲)
- 変数名の軽微な整理 (MATH_SPEC の記号対応表は維持する)

---

## Prohibited Actions

明示的な許可なしに行わない:

- `docs/MATH_SPEC.md` の数式定義を変更する (矛盾発見時は報告のみ)
- MBD の層分離を壊す (estimator/geometry から truth を参照する)
- 乱数 seed を固定しないコードを書く (再現性が壊れる)
- 大量の依存を追加する、フレームワークを勝手に変える
- `git push --force`、依頼なしの push / PR 作成

---

## Definition of Done

- 依頼された機能・修正が実装されている
- `docs/MATH_SPEC.md` の数値テストケースがすべて通る (`pytest tests/test_math_cases.py`)
- `test_roundtrip` / `test_noise_free` / `test_separation` が通る
- MBD 層分離が守られている (estimator/geometry が truth を import していない)
- 乱数を使う箇所は seed で再現可能
- 依頼範囲外の不要な変更がない
- README が必要に応じて更新されている
- 残課題・未確認項目が明示されている

---

## Final Response Format

作業完了時は次の形式で報告してください。

```md
## Summary
- 変更内容1
- 変更内容2

## Verification
- `pytest tests/test_math_cases.py`: passed / failed / not run
- `pytest tests/` (all): passed / failed / not run
- `python scripts/run_minimum.py`: RMSE_xyz = ___ mm (基準: ___ 以下)
- MBD層分離チェック (test_separation): passed / failed

## Notes
- 残課題
- 注意点 (特に MATH_SPEC との対応で迷った箇所)
- 次にやるとよいこと
```

実行できなかった確認項目は理由を明記してください。
