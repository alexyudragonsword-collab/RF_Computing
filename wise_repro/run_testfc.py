"""Port of Hybrid/TestFC.m: random 100x100 complex MVMs through the simulated RF link.

    python run_testfc.py                 # original settings (GetParam_low, easy mode, -37 dBm)
    python run_testfc.py --power 200     # effectively noiseless, checks the port itself
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wise_sim as ws

ap = argparse.ArgumentParser()
ap.add_argument("--power", type=float, default=None, help="override powerRF in dBm")
ap.add_argument("--tests", type=int, default=20)
ap.add_argument("--size", type=int, default=100, help="input and output size")
ap.add_argument("--out", default="results")
ap.add_argument("--tag", default="testfc")
args = ap.parse_args()

rng = np.random.default_rng(0)
p = ws.get_param(low=True)
p["transMode"] = "easy"
p["userNum"] = 1
if args.power is not None:
    p["powerRF"] = args.power

T, N, M = args.tests, args.size, args.size
x = rng.random((T, N)) * np.exp(1j * 2 * np.pi * rng.random((T, N)))
W = rng.random((T, N, M)) * np.exp(1j * 2 * np.pi * rng.random((T, N, M)))
y_dig = np.einsum("tn,tnm->tm", x, W)

out, snr, wave_time = ws.layer_fc(x, W, "auto", p, rng)
y_ana = out[0, 0]

pear = ws.calc_pearson(np.abs(y_dig), np.abs(y_ana))
rmse = ws.calc_rmse(np.abs(y_dig), np.abs(y_ana))
ratio = y_dig / y_ana
phase_spread = float(np.std(np.angle(ratio / ratio.mean(axis=1, keepdims=True))))
res = dict(powerRF_dBm=p["powerRF"], snr_dB=snr, wave_time_s=wave_time,
           pearson_mean=float(pear.mean()), rmse_mean=float(rmse.mean()),
           phase_spread_rad=phase_spread, tests=T, size=N)
print(json.dumps(res, indent=1))

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
    json.dump(res, f, indent=1)
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(np.abs(y_dig), np.abs(y_ana), s=3, alpha=0.4)
ax.set_xlabel("Digital Computing")
ax.set_ylabel("Analog Computing")
ax.set_title(f"RMSE: {rmse.mean():.4f}  Pear: {pear.mean():.4f}  SNR: {snr:.1f} dB")
fig.tight_layout()
fig.savefig(os.path.join(args.out, f"{args.tag}.png"), dpi=130)
