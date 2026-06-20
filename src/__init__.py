"""AquaBeacon Simulation (MBD) package.

MBD の層分離 (truth / sensor / estimator / geometry / evaluation) を守ること。
- truth, sensors: 真値を見てよい
- estimator, geometry: 真値を見てはならない (入力は観測値のみ)
- evaluation: ここでだけ真値 vs 推定を突き合わせる
"""
