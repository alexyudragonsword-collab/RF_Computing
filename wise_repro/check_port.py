"""Checks that the numpy port is faithful, and locates where the simulated error comes from.

1. Discrete check: multiply the two transmit waveforms sample by sample and look for a
   DFT bin that equals x @ W exactly for every symbol. This validates the encoding code.
2. Noiseless runs (powerRF = 200 dBm) through the full simulated link:
   time vs freq encoding, and freq encoding with longer cyclic prefix / padding.
"""
import json
import os

import numpy as np

import wise_sim as ws

out = {}

# 1. discrete product check
rng = np.random.default_rng(1)
p = ws.get_param(low=True)
p.update(transMode="easy", powerRF=200)
T, N, M = 2, 10, 4
x = rng.random((T, N)) * np.exp(2j * np.pi * rng.random((T, N)))
W = rng.random((T, N, M)) * np.exp(2j * np.pi * rng.random((T, N, M)))
yd = np.einsum("tn,tnm->tm", x, W)
in_n, w_n, info = ws.pre_processing(x[None], W, -1, p)
S, Nn, Mn = w_n.shape
wi, ww, _ = ws.generate_tx(in_n, w_n, 1, info["pad"]["subOffset"], (S, Nn, Mn), p)
cp, pad = Nn * ws.iround(Mn * (2 / Mn)), Nn * ws.iround(Mn * p["padRate"])
bl = Nn * Mn + cp + pad
z = (wi[0] * np.conj(ww)).reshape(S, bl)
start = pad // 2 + ws.iround(cp / 2)
Z = np.fft.fft(z[:, start:start + Nn * Mn], axis=1)
exact = []
for k in range(Nn * Mn):
    c = Z[:, k] / yd.reshape(-1)
    if np.std(c) / abs(np.mean(c)) < 1e-9:
        exact.append(k)
out["discrete_exact_bins"] = exact
print("discrete product: bins equal to x@W up to one common factor:", exact)


# 2. full simulated link without noise
def run(enc, cp_rate=-1, pad_rate=0.333, n=100, t=5):
    r = np.random.default_rng(0)
    q = ws.get_param(low=True)
    q.update(transMode="easy", powerRF=200, encMode=enc, cpRate=cp_rate, padRate=pad_rate)
    xx = r.random((t, n)) * np.exp(2j * np.pi * r.random((t, n)))
    ww_ = r.random((t, n, n)) * np.exp(2j * np.pi * r.random((t, n, n)))
    ydd = np.einsum("tn,tnm->tm", xx, ww_)
    ya = ws.layer_fc(xx, ww_, "auto", q, r)[0][0, 0]
    ratio = ydd / ya
    return dict(encMode=enc, cpRate=cp_rate, padRate=pad_rate,
                pearson=float(ws.calc_pearson(abs(ydd), abs(ya)).mean()),
                rmse=float(ws.calc_rmse(abs(ydd), abs(ya)).mean()),
                phase_std_rad=float(np.std(np.angle(ratio / ratio.mean(1, keepdims=True)))))


out["noiseless"] = [run("time"), run("time", 10, 3), run("freq"), run("freq", 4, 1), run("freq", 10, 3)]
for row in out["noiseless"]:
    print(row)

os.makedirs("results", exist_ok=True)
with open("results/port_check.json", "w") as f:
    json.dump(out, f, indent=1)
