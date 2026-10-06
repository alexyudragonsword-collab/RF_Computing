"""Port of Hybrid/Simulation/main_dataset.m: run a trained WISE model through the
simulated RF link at several RF input powers, and compare with digital inference.

    python run_dataset.py --model ML/Result/MNIST_size14_FC3/model_accMax.mat \
                          --data  ML/Data/Data_MNIST_zadoff_14.mat
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import wise_sim as ws

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--samples", type=int, default=1000, help="test samples for the analog run")
ap.add_argument("--powers", default="-70:-2:-110", help="start:step:stop in dBm, MATLAB style")
ap.add_argument("--mode", default="fast", choices=["fast", "easy"])
ap.add_argument("--encmode", default="time", choices=["time", "freq"])
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--out", default="results")
ap.add_argument("--tag", default="dataset_mnist_fc3")
ap.add_argument("--name", default=None, help="model name for the report, default is the model folder name")
args = ap.parse_args()

a, s, b = (float(v) for v in args.powers.split(":"))
POWERS = list(np.arange(a, b + s / 2, s)) if s < 0 else list(np.arange(a, b + s / 2, s))
POWER_LO = -3.56
ACT, METHOD, CLASSES, BATCH = "zadoff", "auto", 10, 100


def make_param():
    p = ws.get_param()
    p.update(encMode=args.encmode, decMode="split-4", transMode=args.mode, sampleRate=100e6,
             subMax=float("inf"), subMin=1, cpRate=-1,
             guardInput=1e-10, guardOutput=1e-10, guardDC=0,
             insertion=0, noisefigure=0)
    return p


model = loadmat(args.model)
fc_list = []
k = 1
while f"Matrix_{k}" in model:
    fc_list.append(model[f"Matrix_{k}"].astype(complex).T)    # FCList{k} = Matrix_k.'
    k += 1
mac_num = sum(fc.shape[0] * fc.shape[1] for fc in fc_list)
data = loadmat(args.data)
X = data["dataList"].astype(complex)
Y = data["targetList"].reshape(-1).astype(int)

pred_dig = np.array([np.argmax(ws.digital_inference(x, fc_list, ACT)) for x in X])
acc_dig = float(np.mean(pred_dig == Y))
print(f"layers {[fc.shape[::-1] for fc in fc_list]}  MACs {mac_num}  digital acc {acc_dig:.4f} on {len(Y)}")

n = min(args.samples, len(Y))


def run_power(power):
    rng = np.random.default_rng(int(abs(power) * 1000) + 7)
    p = make_param()
    p["powerRF"], p["powerLO"] = power, POWER_LO
    correct, wave_time = 0, 0.0
    agree = 0
    for bi in range(int(math.ceil(n / BATCH))):
        s0, e0 = bi * BATCH, min((bi + 1) * BATCH, n)
        out, wave_time = ws.analog_inference(X[s0:e0], fc_list, ACT, METHOD, p, rng)
        pred = np.argmax(out[0], axis=1)
        correct += int(np.sum(pred == Y[s0:e0]))
        agree += int(np.sum(pred == pred_dig[s0:e0]))
    return power, correct / n, agree / n, wave_time


with ProcessPoolExecutor(max_workers=args.workers) as ex:
    rows = sorted(ex.map(run_power, POWERS), key=lambda r: -r[0])

res = dict(model=args.name or os.path.basename(os.path.dirname(os.path.abspath(args.model))), mode=args.mode, encMode=args.encmode,
           samples_analog=n, samples_digital=len(Y), acc_digital=acc_dig,
           acc_digital_on_subset=float(np.mean(pred_dig[:n] == Y[:n])), mac_num=mac_num,
           powerLO_dBm=POWER_LO, rows=[])
for power, acc, agree, wt in rows:
    emac = 10 ** (power / 10) / 1000 * wt / mac_num / BATCH
    res["rows"].append(dict(powerRF_dBm=float(power), acc_analog=acc, agree_with_digital=agree,
                            emac_J=emac, wave_time_s=wt))
    print(f"P_RF {power:7.1f} dBm  E_MAC {emac:.2e} J  acc {acc:.4f}  agrees with digital {agree:.4f}")

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
    json.dump(res, f, indent=1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([r["emac_J"] for r in res["rows"]], [1 - r["acc_analog"] for r in res["rows"]], "+-", label="analog (simulated)")
ax.axhline(1 - res["acc_digital_on_subset"], color="gray", ls="--", label="digital")
ax.set_xscale("log")
ax.set_xlabel("E_MAC  J/MAC")
ax.set_ylabel("Error rate")
ax.set_ylim(0, 1)
ax.set_title(f"{res['model']}  ({n} test samples, {args.mode} mode)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(args.out, f"{args.tag}.png"), dpi=130)
