"""results_io.py — 実行結果を機械可読 (JSON / CSV) で保存するユーティリティ。

print だけだと後から解析・比較できないため、run_*.py の数値結果を results/ に
書き出す。MBD 上は評価/出力の補助 (truth/estimator のロジックには関与しない)。

- write_json(name, payload): results/<name>.json に辞書を保存 (メタ情報付き)。
- write_csv(name, rows, header): results/<name>.csv に表を保存。

再現性のため、保存する payload には呼び出し側で seed や設定を含めること。
"""
import csv
import json
import os
from datetime import datetime, timezone

# results/ はリポジトリ直下 (このファイルは src/ にある)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _stamp():
    """ISO8601 (UTC) のタイムスタンプ文字列。記録の追跡用。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(name, payload, meta=None):
    """results/<name>.json に payload を保存する。

    payload : JSON 化できる辞書 (numpy はあらかじめ float/list に変換しておく)。
    meta    : 追加メタ情報 (seed・設定など)。生成時刻は自動で付与する。
    戻り値  : 保存先パス。
    """
    _ensure_dir()
    record = {"generated_at": _stamp(), "meta": meta or {}, "result": payload}
    path = os.path.join(RESULTS_DIR, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def write_csv(name, rows, header=None):
    """results/<name>.csv に rows (list[list] or list[dict]) を保存する。

    header を渡すと先頭行に書く。rows が dict のリストなら header をキー順に使う。
    戻り値: 保存先パス。
    """
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, name + ".csv")
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
