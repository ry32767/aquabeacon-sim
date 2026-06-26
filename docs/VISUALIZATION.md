# VISUALIZATION.md — 可視化仕様 (発表・デモ用)

推定結果を**人に見せる**ための図・アニメーションの仕様。
生成は `scripts/run_visualize.py`、出力は `results/visualize/` 以下
(他シナリオと同様に `results/<シナリオ>/` へ統合。自動生成の `README.md` 付き)。

## 出力フォルダ構成 (テストシナリオごとに分割)

`results/visualize/` は**シーンごとにサブフォルダ**へ分けて出力する。
トップ階層は本番ハードの2系統 (測位=親機1カメラ / ジオメトリ=子機ステレオ) でグループ化:

```
results/visualize/
├── README.md                 # 自動生成 (条件スナップショット + シーン一覧)
├── positioning/              # 親機カメラ(角度) + 音響(距離) による位置・軌道推定
│   ├── 1_cloud3d/            # scene1  cloud3d.png + cloud3d_rotate.gif/.mp4
│   ├── 2_sensitivity/        # scene2  sensitivity.png
│   ├── 3_converge/           # scene3  converge.png/.gif/.mp4
│   ├── 4_trajectory/         # scene4  trajectory.png/.gif/.mp4
│   ├── 5_traj_imu/           # scene5  traj_imu.png + traj_imu_rotate.gif/.mp4
│   ├── 7_mapping_progress/   # scene7  mapping_progress.png/.gif/.mp4
│   └── 9_traj_converge/      # scene9  traj_converge.png/.gif/.mp4
├── geometry/                 # 子機ステレオ2カメラ による3D計測 (MATH_SPEC §6.2)
│   ├── 6_cube_mapping/       # scene6  cube_mapping.png + cube_mapping_rotate.gif/.mp4
│   ├── 8_multilook_converge/ # scene8  multilook_converge.png/.gif/.mp4
│   └── 10_stage2_sensitivity/# scene10 stage2_sensitivity.png
└── validation/               # 研究グレードの不確かさ・効率・幾何希釈 (MATH_SPEC §4.5, §15)
    ├── 11_crlb_ellipsoid/    # scene11 crlb_ellipsoid.png + crlb_ellipsoid_rotate.gif/.mp4
    ├── 12_gdop_map/          # scene12 gdop_map.png
    ├── 13_efficiency/        # scene13 efficiency.png
    └── 14_fusion_uncertainty/# scene14 fusion_uncertainty.png
```

各フォルダに 1 シナリオの PNG (+ アニメは GIF / MP4) がまとまる。発表時はフォルダ単位で扱える。

## MBD 上の位置づけ

可視化は**評価/プレゼン層**であり、真値と推定値を突き合わせてよい (evaluation と同じ扱い)。
ただし推定 (`estimate_position`) には truth を渡さず、ノイズ付き観測のみを入力する原則は守る。
推定アルゴリズム自体 (`src/estimator.py`) は truth を一切参照しない (test_separation が強制)。

## 実行

```bash
python scripts/run_visualize.py
```

- 乱数はすべて seed 固定 (再現可能)。
- 日本語フォント (Yu Gothic / Meiryo / MS Gothic / Noto Sans CJK 等) を自動検出し、ラベルを日本語表示する。
  見つからない場合は英語ラベルに自動フォールバックして文字化け (豆腐□) を防ぐ。
- アニメは **GIF を必ず出力**し、`ffmpeg` があれば **MP4 も追加出力**する。

## 生成物 (4 シーン)

| シーン | 内容 | 出力フォルダ / ファイル |
|--------|------|-------------|
| **scene1: 3D 推定クラウド** | 親機(原点)・子機真値・モンテカルロ推定点群 (N=1500) を 3D 散布。推定のばらつきを **2σ 誤差楕円体**で重ねる。回転ビューで立体感を見せる。 | `positioning/1_cloud3d/` : `cloud3d.png`, `cloud3d_rotate.gif`, `.mp4` |
| **scene2: 感度解析グラフ** | (a) RMSE vs 距離 d (理論線 `d·σ_ang` 重ね)、(b) RMSE vs 角度ノイズ、(c) RMSE vs 仰角 φ (真下 rho≈0 付近の破綻チェック)。`run_sensitivity.py` の表をグラフ化。 | `positioning/2_sensitivity/` : `sensitivity.png` |
| **scene3: 最小二乗の収束** | 初期値 (真値から 5m ずらす) からガウス・ニュートン反復 (MATH_SPEC §4.2) で推定点が真値へ収束する過程。「推定している」ことを可視化。ノイズフリーなので数反復で誤差≈0。 | `positioning/3_converge/` : `converge.png`, `.gif`, `.mp4` |
| **scene4: 軌道追従 (Stage 2 先取り)** | 子機が芝刈り (boustrophedon) 軌道を動き、各時刻で Stage 1 の単時刻推定を独立適用。真の軌道(赤)を推定(青)が追従する様子をアニメ化。マッピングのイメージを先取りで見せる。 | `positioning/4_trajectory/` : `trajectory.png`, `.gif`, `.mp4` |
| **scene5: 軌道推定 IMU有無 (Stage 2)** | ダブル芝刈り軌道を「観測のみ」と「観測+IMU拘束」で推定し比較 (MATH_SPEC §5)。IMU拘束で推定点(青)が真の軌道に密着し、RMSE が下がる (例 96→34 mm) ことを可視化。 | `positioning/5_traj_imu/` : `traj_imu.png`, `traj_imu_rotate.gif`, `.mp4` |
| **scene6: キューブ計測 (子機ステレオ, Stage 2)** | 既知キューブ表面の真点群(赤) と、**子機の2カメラ(ステレオ)三角測量**を多フレーム平均して復元した推定点群(青) を重ねる (MATH_SPEC §6.2)。寸法誤差・体積誤差率・点群RMS をタイトルに表示。 | `geometry/6_cube_mapping/` : `cube_mapping.png`, `cube_mapping_rotate.gif`, `.mp4` |
| **scene7: マッピング進行アニメ (IMU有無)** | 子機がダブル芝刈り軌道を進むにつれ、IMUなし(灰)とIMUあり(青)の推定が時間とともに現れる (MATH_SPEC §5)。IMUありが真の軌道に密着して進む様子を時間進行で見せる。 | `positioning/7_mapping_progress/` : `mapping_progress.png`, `.gif`, `.mp4` |
| **scene8: ステレオ多フレーム平均の収束アニメ** | 撮影フレーム数 looks=1→30 を増やすにつれ、子機ステレオで復元したキューブ点群(青) が真の表面(赤) に締まり、寸法誤差・体積誤差率・点群RMS が改善する過程をアニメ化 (MATH_SPEC §6.2)。「多フレーム平均が精度のレバー」を可視化。 | `geometry/8_multilook_converge/` : `multilook_converge.png`, `.gif`, `.mp4` |
| **scene9: 軌道推定の収束アニメ** | 散らかった初期軌道から、複数時刻バンドル調整 (ガウス・ニュートン反復, MATH_SPEC §5) で軌道全体が真値へ収束していく過程。ノイズフリーなので数反復で RMSE≈0。 | `positioning/9_traj_converge/` : `traj_converge.png`, `.gif`, `.mp4` |
| **scene10: Stage 2 ステレオ感度グラフ** | (a) 観測距離 standoff、(b) ベースライン B、(c) 撮影フレーム数 looks を振ったときの寸法誤差・体積誤差率・点群RMS (静止画, 3パネル, 2軸)。**子機を近づけるほど・ベースラインを長くするほど・多フレームほど精度が上がる**ステレオ設計の根拠 (MATH_SPEC §6.2)。 | `geometry/10_stage2_sensitivity/` : `stage2_sensitivity.png` |
| **scene11: CRLB 楕円体の重ね描き** | 経験推定クラウド + 経験 2σ 楕円体に、**解析 CRLB 2σ 楕円体 (緑ワイヤ)** を重ねる。経験のばらつきが理論下界とほぼ一致する = **推定が効率的 (情報理論的に最適)** を立体的に見せる (MATH_SPEC §4.5, §15)。回転 GIF/MP4 つき。 | `validation/11_crlb_ellipsoid/` : `crlb_ellipsoid.png`, `.gif`, `.mp4` |
| **scene12: GDOP マップ** | 距離 d × 仰角 φ の格子で **GDOP (位置 1σ 半径)** をヒートマップ表示し、等高線 (50/75/100/150/200 mm) を重ねる。**観測幾何 → 達成可能精度**の地図で、near-nadir 運用域の根拠 (MATH_SPEC §4.5)。 | `validation/12_gdop_map/` : `gdop_map.png` |
| **scene13: 効率 (経験RMSE→CRLB)** | (a) 仰角掃引・(b) 距離掃引で、経験モンテカルロ RMSE が **CRLB に漸近**する様子を **95% 信頼区間つき**で示す (MATH_SPEC §15)。論文の中核検証図をそのまま発表に使える。 | `validation/13_efficiency/` : `efficiency.png` |
| **scene14: センサ構成と CRLB (正直比較)** | 同一 near-nadir 幾何で 光学+音響1点 / +深度 / SBL4点(4m)+深度 / SBL4点(16m)+深度 の **位置1σ (GDOP) と z 1σ** を棒グラフ比較。**深度→z圧縮**、**SBL は光なしだが深い子機では大型アレイ要** (§13.2) という正直な設計知見を一目で示す。 | `validation/14_fusion_uncertainty/` : `fusion_uncertainty.png` |

> scene7〜10 は Stage 2 の「動的アニメ + 感度」。scene5/6 (回転ビュー) と合わせて Stage 2 の理解を立体的に見せる。
> 各シーンとも乱数は seed 固定で再現可能。推定/三角測量 (estimate_trajectory / stereo_triangulate) には truth を渡さず観測のみを入力する。
> **scene6/8/10 のジオメトリは「子機の2カメラ(ステレオ)三角測量」** (MATH_SPEC §6.2)。位置推定 (scene1〜5,7,9) は親機1カメラ+音響。

> scene4 の軌道は `src/truth.demo_trajectory` が生成する**発表デモ用の簡易芝刈り**。
> 本格的な Stage 2 のダブル芝刈り + IMU 拘束 (MATH_SPEC §5) はマッピング段階で正式実装する。
> 各時刻の推定は単時刻 (Stage 1) を独立適用しているだけで、まだ時刻間拘束は入っていない。

## 依存

- `matplotlib` (必須。3D 描画・アニメ)
- `pillow` (GIF 書き出し。`PillowWriter`)
- `ffmpeg` (任意。MP4 書き出し。無ければ GIF のみ)

## 数値の妥当性 (発表時の説明用)

- 3D クラウドの RMSE total ≈ 90 mm @ d=12.5m, σ_ang=0.3° は、理論オーダー `d·σ_ang ≈ 65 mm`
  (横方向) と距離ノイズ 30 mm の合成として妥当 (MATH_SPEC §7)。
- 感度グラフ (a) で RMSE が距離 d に比例して増えること、(c) で真下付近でも破綻しないことが確認できる。
