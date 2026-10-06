"""Build gui/demo_mnist.npz from the test set that WISE's ML/main.py writes, and
gui/model_fc3.npz from results/model/model_accMax.mat, so the GUI needs no scipy.

    python gui/make_demo_data.py <WISE>/ML/Data/Data_MNIST_zadoff_14.mat
"""
import os
import sys

import numpy as np
from scipy.io import loadmat

src = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
d = loadmat(src)
x = d["dataList"][:n].astype(np.complex128)
y = d["targetList"].reshape(-1)[:n].astype(np.int64)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_mnist.npz")
np.savez_compressed(out, x=x, y=y)
print(out, x.shape, np.bincount(y, minlength=10))

here = os.path.dirname(os.path.abspath(__file__))
m = loadmat(os.path.join(os.path.dirname(here), "results", "model", "model_accMax.mat"))
mats = {k: m[k] for k in m if k.startswith("Matrix_")}
np.savez_compressed(os.path.join(here, "model_fc3.npz"), **mats)
print("model_fc3.npz", {k: v.shape for k, v in mats.items()})
