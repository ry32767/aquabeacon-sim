"""results_io.py — シナリオ結果を results/<scenario>/ に統合保存するユーティリティ。

各シナリオ (run_*.py) の出力 (図 PNG/GIF/MP4・数値 JSON/CSV・説明 README.md) を
results/<scenario>/ にまとめ、シナリオ単位で結果を一覧できるようにする。
MBD 上は評価/出力の補助 (truth/estimator のロジックには関与しない)。

- scenario_dir(name)         : results/<name>/ を作って返す (図・データの保存先)。
- write_json(name, payload)  : results/<name>.json に辞書を保存 (name は "scenario/file" 可)。
- write_csv(name, rows)      : results/<name>.csv に表を保存 (同上)。
- write_report(...)          : results/<scenario>/README.md にシナリオ説明を自動生成。
                               実行条件は config.toml のスナップショットを埋め込む。

再現性のため、保存する payload には呼び出し側で seed や設定を含めること。
"""
import csv
import json
import os
from datetime import datetime, timezone

# results/ はリポジトリ直下 (このファイルは src/ にある)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path) or RESULTS_DIR, exist_ok=True)


def _stamp():
    """ISO8601 (UTC) のタイムスタンプ文字列。記録の追跡用。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scenario_dir(name):
    """results/<name>/ を作成して絶対パスを返す (シナリオの図・データ・説明の保存先)。"""
    d = os.path.join(RESULTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(name, payload, meta=None):
    """results/<name>.json に payload を保存する (name は "scenario/file" のサブパス可)。

    payload : JSON 化できる辞書 (numpy はあらかじめ float/list に変換しておく)。
    meta    : 追加メタ情報 (seed・設定など)。生成時刻は自動で付与する。
    戻り値  : 保存先パス。
    """
    record = {"generated_at": _stamp(), "meta": meta or {}, "result": payload}
    path = os.path.join(RESULTS_DIR, name + ".json")
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def write_csv(name, rows, header=None):
    """results/<name>.csv に rows (list[list] or list[dict]) を保存する (サブパス可)。

    header を渡すと先頭行に書く。rows が dict のリストなら header をキー順に使う。
    戻り値: 保存先パス。
    """
    path = os.path.join(RESULTS_DIR, name + ".csv")
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        if rows and isinstance(rows[0], dict):
            keys = header or list(rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        else:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(rows)
    return path


# ----------------------------------------------------------------------------
# シナリオ説明 README.md の自動生成 (config.toml の条件スナップショット付き)
# ----------------------------------------------------------------------------
def _config_raw():
    """config.toml の生辞書と読込状態を返す (循環 import 回避のため遅延 import)。"""
    try:
        from src import config
        return dict(getattr(config, "CONFIG_RAW", {}) or {}), \
            bool(getattr(config, "CONFIG_TOML_LOADED", False))
    except Exception:
        return {}, False


def format_conditions(section_names):
    """config.toml の指定セクションを Markdown 表のスナップショットにする。

    section_names : 関連する [section] 名のリスト。無いセクションは「デフォルト使用」と注記。
    """
    raw, loaded = _config_raw()
    out = []
    if not loaded:
        out.append("> config.toml は未読込 (Python 3.10 等)。`src/config.py` の"
                   "デフォルト値で実行されています。\n")
    for sec in section_names:
        data = raw.get(sec)
        if data:
            out.append(f"**`[{sec}]`**\n")
            out.append("| パラメータ | 値 |")
            out.append("|---|---|")
            for k, v in data.items():
                out.append(f"| `{k}` | `{v}` |")
            out.append("")
        else:
            out.append(f"**`[{sec}]`** — config.toml に記載なし → "
                       f"`src/config.py` のデフォルト値を使用\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# センサ登録: 各シナリオが使うセンサを一元管理し、説明 README と索引にまとめる
# ----------------------------------------------------------------------------
# センサの正準名 (key -> 表示名)。
SENSOR_NAMES = {
    "parent_cam": "親機カメラ (光学角度: 方位/仰角)",
    "acoustic1": "音響測距 (親機1点までの距離)",
    "sbl": "SBL音響 (親機4トランスデューサへの距離=多辺測量)",
    "stereo": "子機ステレオ2カメラ (三角測量)",
    "imu": "IMU (時刻間変位)",
    "depth": "深度センサ (圧力→鉛直 z)",
}
SENSOR_ORDER = ["parent_cam", "acoustic1", "sbl", "stereo", "imu", "depth"]
_SENSOR_SHORT = {"parent_cam": "親機カメラ", "acoustic1": "音響1点",
                 "sbl": "SBL音響", "stereo": "ステレオ", "imu": "IMU", "depth": "深度"}

# シナリオ -> {使用センサ, 一言概要, 対応 MATH_SPEC}。索引と各 README の「使用センサ」に使う。
SCENARIO_INFO = {
    "mapping": {"sensors": ["parent_cam", "acoustic1", "imu", "stereo"],
                "one": "Stage2 軌道推定(IMU) + 子機ステレオでキューブ計測", "spec": "§5,§6.2"},
    "spec": {"sensors": ["parent_cam", "acoustic1", "stereo", "depth"],
             "one": "設計スペック逆算 + 深度センサ込みの最大運用水深", "spec": "§7-§10"},
    "robust": {"sensors": ["parent_cam", "acoustic1", "imu"],
               "one": "ロバスト推定で外れ値(見失い/マルチパス)に耐える", "spec": "§4.4"},
    "deepwater": {"sensors": ["parent_cam", "acoustic1", "imu"],
                  "one": "水中の光減衰で深い/濁るほど精度劣化・見失い", "spec": "§9"},
    "depth": {"sensors": ["parent_cam", "acoustic1", "depth"],
              "one": "深度センサ融合で z 精度↑・単時刻ロバスト", "spec": "§10"},
    "no_optical": {"sensors": ["acoustic1", "imu", "depth"],
                   "one": "光学なし: 距離+IMU+深度で測位 (濁り非依存)", "spec": "§11"},
    "sbl": {"sensors": ["sbl", "imu", "depth"],
            "one": "SBL: 親機4点音響の多辺測量 (光学なし比較手法)", "spec": "§13"},
    "opmap": {"sensors": ["parent_cam", "acoustic1", "imu", "depth"],
              "one": "濁り×水深の運用可能領域マップ (光学/フォールバック/不可)", "spec": "§9-§12"},
    "switch": {"sensors": ["parent_cam", "acoustic1", "imu", "depth"],
               "one": "光学↔フォールバック自動切替 (プルーム通過)", "spec": "§12"},
    "visualize": {"sensors": ["parent_cam", "acoustic1", "imu", "stereo"],
                  "one": "発表用 可視化シーン集 (Stage1 + Stage2)", "spec": "§1-§6.2"},
}


def _sensors_section(scenario, sensors=None):
    keys = sensors if sensors is not None else \
        SCENARIO_INFO.get(scenario, {}).get("sensors", [])
    if not keys:
        return ""
    lines = ["## 使用センサ", ""]
    for k in keys:
        lines.append(f"- {SENSOR_NAMES.get(k, k)}")
    lines.append("")
    return "\n".join(lines)


def write_index():
    """results/README.md に全シナリオの「使用センサ一覧」マトリクスを生成する。

    SCENARIO_INFO を単一の出典として、シナリオ×センサの ✓ 表と各フォルダへのリンクを書く。
    どのシナリオを実行しても (write_report 経由で) 最新に保たれる。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    head = ["# AquaBeacon シミュレーション 結果インデックス", "",
            f"> 自動生成: {_stamp()}", "",
            "各シナリオの結果は `results/<シナリオ>/` にあり、フォルダ内 `README.md` に"
            "詳細 (条件・主な結果・生成物) がある。`results/` は再生成可能 (.gitignore 済み)。",
            "", "## 使用センサ一覧", "",
            "親機=水上の基準局、子機=水中の移動体。各シナリオがどのセンサを使うか:", ""]
    header = "| シナリオ | " + " | ".join(_SENSOR_SHORT[k] for k in SENSOR_ORDER) + " | 概要 |"
    sep = "|---|" + "".join(":-:|" for _ in SENSOR_ORDER) + "---|"
    rows = []
    for name, info in SCENARIO_INFO.items():
        used = set(info.get("sensors", []))
        marks = " | ".join("✓" if k in used else "" for k in SENSOR_ORDER)
        rows.append(f"| [{name}](./{name}/) | {marks} | {info.get('one','')} |")
    legend = ["", "### センサの説明", ""]
    for k in SENSOR_ORDER:
        legend.append(f"- **{_SENSOR_SHORT[k]}**: {SENSOR_NAMES[k]}")
    body = head + [header, sep] + rows + legend + [""]
    path = os.path.join(RESULTS_DIR, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body))
    return path


def write_report(scenario, title, summary, condition_sections=(),
                 outputs=(), results=None, meta=None, math_spec=None, sensors=None):
    """results/<scenario>/README.md にシナリオ説明を自動生成する。

    scenario          : シナリオ名 (results/<scenario>/ に書く)。
    title             : 見出し。
    summary           : このシナリオが何か・何を見るかの説明 (Markdown 文字列)。
    condition_sections: 実行条件として埋め込む config.toml のセクション名リスト。
    outputs           : (ファイル名, 説明) のリスト。生成物への相対リンクを作る。
    results           : 主な数値結果の dict (任意, 表示用)。
    meta              : seed 等の補足 dict。
    math_spec         : 対応する MATH_SPEC 節 (例 "§9, §12")。
    戻り値            : README.md のパス。
    """
    d = scenario_dir(scenario)
    lines = [f"# {title}", ""]
    tags = [f"自動生成: {_stamp()}"]
    if math_spec:
        tags.append(f"MATH_SPEC: {math_spec}")
    if meta:
        tags += [f"{k} = {v}" for k, v in meta.items()]
    lines.append("> " + " / ".join(tags))
    lines.append("")

    lines.append("## このシナリオは何か")
    lines.append("")
    lines.append(summary.strip())
    lines.append("")

    sec = _sensors_section(scenario, sensors)
    if sec:
        lines.append(sec)

    if results:
        lines.append("## 主な結果")
        lines.append("")
        lines.append("| 指標 | 値 |")
        lines.append("|---|---|")
        for k, v in results.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if condition_sections:
        lines.append("## 実行条件 (config.toml スナップショット)")
        lines.append("")
        lines.append("この結果を生成したときの主要パラメータ。`config.toml` を編集して"
                     "再実行すると条件が変わる。")
        lines.append("")
        lines.append(format_conditions(list(condition_sections)))
        lines.append("")

    if outputs:
        lines.append("## 生成物")
        lines.append("")
        for item in outputs:
            if isinstance(item, (list, tuple)):
                fname, desc = item[0], (item[1] if len(item) > 1 else "")
            else:
                fname, desc = item, ""
            lines.append(f"- [`{fname}`](./{fname})" + (f" — {desc}" if desc else ""))
        lines.append("")

    path = os.path.join(d, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    write_index()                      # 全シナリオのセンサ索引 (results/README.md) を更新
    return path
