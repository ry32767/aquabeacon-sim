"""pytest 設定: リポジトリルートを sys.path に入れ、`from src.xxx import ...` を可能にする。

pytest は conftest.py のあるディレクトリ (= リポジトリルート) を sys.path に追加するため、
本ファイルが存在するだけで tests/ から src パッケージを import できる。

加えて scripts/ も sys.path に追加する。test_spec が `from scripts.run_spec import ...` で
スクリプトを読み込むと、その中の `from _plotstyle import ...` (scripts/ 直下の共有描画
モジュール) を解決する必要があるため。スクリプトを直接実行する場合は scripts/ が自動で
sys.path[0] に入るので、この追加はテスト経由の import 専用。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
