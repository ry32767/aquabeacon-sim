"""pytest 設定: リポジトリルートを sys.path に入れ、`from src.xxx import ...` を可能にする。

pytest は conftest.py のあるディレクトリ (= リポジトリルート) を sys.path に追加するため、
本ファイルが存在するだけで tests/ から src パッケージを import できる。
"""
