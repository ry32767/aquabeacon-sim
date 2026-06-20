"""
test_separation.py — MBD の層分離を機械的に強制する。

鉄則: estimator と geometry は truth を絶対に参照しない (入力はセンサ値のみ)。
このテストは src/estimator.py と src/geometry.py のソースを静的に読み、
truth を import していないことを確認する。

src がまだ無い段階ではスキップされる (存在しないファイルは検査対象外)。
"""
import os
import ast

SRC = os.path.join(os.path.dirname(__file__), "..", "src")

# 推定・ジオメトリ層が import してはいけないモジュール名
FORBIDDEN = {"truth", "src.truth"}
# 検査対象 (真値を見てはいけない層)
GUARDED = ["estimator.py", "geometry.py"]


def _imports_of(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_estimator_and_geometry_do_not_import_truth():
    for fname in GUARDED:
        path = os.path.join(SRC, fname)
        if not os.path.exists(path):
            continue  # その段階に未到達ならスキップ
        imported = _imports_of(path)
        leaked = imported & FORBIDDEN
        assert not leaked, (
            f"{fname} が truth を参照しています: {leaked}. "
            f"MBDの層分離違反。推定/ジオメトリは観測値のみを入力にすること。"
        )


if __name__ == "__main__":
    test_estimator_and_geometry_do_not_import_truth()
    print("separation OK")
