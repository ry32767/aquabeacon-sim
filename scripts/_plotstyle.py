"""_plotstyle.py — 各 run_*.py が共有する matplotlib 描画スタイルの一元管理。

これまで各スクリプトに同じ「Agg バックエンド設定 + 日本語フォント検出 + Lbl」が
コピーされていた。フォント候補や凡例方針を変えるたびに全スクリプトを直す必要が
あったため、ここに集約する (MBD の src/ ではなく描画専用なので scripts/ 配下)。

使い方:
    from _plotstyle import plt, USE_JP, Lbl

公開シンボル:
    plt     : 設定済みの matplotlib.pyplot (Agg バックエンド・日本語フォント適用済み)
    USE_JP  : 日本語フォントが見つかったか (bool)。
    JP_FONT : 採用した日本語フォント名 (str)。見つからなければ None。
    Lbl     : Lbl(ja, en) — 日本語フォントがあれば ja、無ければ en を返す。
"""
import matplotlib
matplotlib.use("Agg")                      # 画面なしでファイル保存するため
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント候補 (Windows / macOS / Linux の順に試す)。最初に見つかったものを使う。
_JP_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                  "Hiragino Sans", "TakaoPGothic", "IPAexGothic"]
_available = {f.name for f in fm.fontManager.ttflist}
JP_FONT = next((c for c in _JP_CANDIDATES if c in _available), None)
USE_JP = JP_FONT is not None
if USE_JP:
    plt.rcParams["font.family"] = JP_FONT
plt.rcParams["axes.unicode_minus"] = False


def Lbl(ja, en):
    """日本語フォントがあれば ja、無ければ en を返す (文字化け回避のフォールバック)。"""
    return ja if USE_JP else en
