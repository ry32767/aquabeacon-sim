"""rng.py — 再現可能で**独立な**乱数ストリームの生成 (MATH_SPEC §15.2)。

モンテカルロの統計的妥当性には、各試行・各センサ・各時刻のノイズが互いに**独立**である
ことが要る。`base_seed + s` を試行ごとに 1 ずつ増やし、内部で `seed + k` を時刻に使う従来
方式は、隣接試行が同じ per-step ストリームを 1 ステップずらして再利用するため**試行が独立に
ならない** (過小分散・相関した RMSE)。本モジュールは `numpy.random.SeedSequence` で互いに素な
ストリームを導出し、これを構造的に防ぐ。

MBD 上は中立な乱数ユーティリティ (truth も推定も含まない)。
"""
import numpy as np


def substream_seed(base_seed, *keys):
    """(base_seed, *keys) ごとに**よく離れた** 64bit 整数シードを返す (MATH_SPEC §15.2)。

    SeedSequence([base, *keys]) から導出するので、(試行, センサ, …) の組ごとに統計的に独立な
    整数になる。これを各 *_sequence 生成器の `seed=` に渡せば、内部の `seed+k` (時刻デコリレ)
    の範囲が試行間・センサ間で衝突しない (連続加算による再利用を回避)。

    例: 試行 s・センサ obs(0)/imu(1)/depth(2) で
        seed_obs   = substream_seed(SEED, s, 0)
        seed_imu   = substream_seed(SEED, s, 1)
        seed_depth = substream_seed(SEED, s, 2)
    """
    ss = np.random.SeedSequence([int(base_seed)] + [int(k) for k in keys])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def spawn_generators(base_seed, n):
    """base_seed から互いに独立な Generator を n 個生成して返す (list)  (MATH_SPEC §15.2)。

    np.random.SeedSequence(base_seed).spawn(n) による厳密に独立なサブストリーム。
    連続ストリームでノイズを引きたい場合 (時間相関 §8.6 等) に各試行へ1本ずつ渡す。
    """
    ss = np.random.SeedSequence(int(base_seed))
    return [np.random.default_rng(child) for child in ss.spawn(int(n))]
