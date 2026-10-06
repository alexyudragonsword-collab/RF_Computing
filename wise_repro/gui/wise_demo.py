"""WISE 射频计算演示程序（PySide6）

    python gui/wise_demo.py
    python gui/wise_demo.py --screenshots shots/     # 离屏生成三页截图后退出

三个页面：
  原理演示   频域编码 + 混频 = 矩阵向量乘，可调 N、M，点选读出窗口里的 y
  单层仿真   TestFC：随机复数矩阵走一遍模拟射频链路，对比数字结果
  MNIST 推理 训练好的复数网络逐层走模拟链路，拖动射频功率看准确率变化
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.io import loadmat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import wise_sim as ws  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "results", "model", "model_accMax.mat")
CURVE_PATH = os.path.join(ROOT, "results", "dataset_mnist_fc3.json")
DEMO_PATH = os.path.join(HERE, "demo_mnist.npz")

C_X, C_W, C_Y = "#1d64a8", "#a4620f", "#b4185a"
C_FG, C_MUTED, C_RULE, C_BG = "#16202a", "#7d8a97", "#d3dbe2", "#f3f5f7"
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

matplotlib.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC",
                        "WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C_RULE, "axes.labelcolor": C_FG, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

QSS = f"""
QMainWindow, QWidget#page {{ background: {C_BG}; }}
QGroupBox {{ background: white; border: 1px solid {C_RULE}; border-radius: 8px; margin-top: 14px;
            padding: 10px 10px 8px 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {C_FG}; }}
QLabel#lead {{ color: #46525e; font-size: 13px; }}
QLabel#metric {{ background: white; border: 1px solid {C_RULE}; border-radius: 6px; padding: 6px 10px; }}
QPushButton {{ background: white; border: 1px solid #9aa8b5; border-radius: 6px; padding: 5px 14px; }}
QPushButton:hover {{ border-color: {C_FG}; }}
QPushButton#primary {{ background: {C_X}; color: white; border-color: {C_X}; font-weight: 600; }}
QPushButton#primary:disabled {{ background: #9fb7cf; border-color: #9fb7cf; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ padding: 8px 18px; font-size: 13px; }}
QTabBar::tab:selected {{ color: {C_X}; font-weight: 600; border-bottom: 2px solid {C_X}; }}
"""


# ----------------------------------------------------------------------------- helpers

class Signals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)


class Job(QtCore.QRunnable):
    """Runs fn(*args, progress=...) on the thread pool and reports back through Qt signals."""

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args, self.signals = fn, args, Signals()

    def run(self):
        try:
            res = self.fn(*self.args, progress=self.signals.progress.emit)
        except Exception as e:  # report instead of killing the worker thread
            self.signals.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.signals.done.emit(res)


class Plot(FigureCanvasQTAgg):
    def __init__(self, w=6, h=4):
        self.fig = Figure(figsize=(w, h), layout="constrained")
        super().__init__(self.fig)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)


def spin(lo, hi, val, step=1):
    w = QtWidgets.QSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    return w


def slider(lo, hi, val):
    w = QtWidgets.QSlider(Qt.Orientation.Horizontal)
    w.setRange(lo, hi)
    w.setValue(val)
    return w


def lead(text):
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("lead")
    lab.setWordWrap(True)
    return lab


def mono():
    f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    f.setPointSize(10)
    return f


def fmt_c(z):
    re, im = float(np.real(z)), float(np.imag(z))
    return f"{re:+.3f} {'−' if im < 0 else '+'} {abs(im):.3f}i".replace("-", "−")


def stems(ax, idx, h, color, alpha=1.0, lw=2.0, ms=5):
    ax.vlines(idx, 0, h, color=color, lw=lw, alpha=alpha)
    ax.plot(idx, h, "o", color=color, ms=ms, alpha=alpha)


def mnist_param(power, mode, enc):
    """Parameters of Hybrid/Simulation/main_dataset.m."""
    p = ws.get_param()
    p.update(encMode=enc, decMode="split-4", transMode=mode, sampleRate=100e6,
             subMax=float("inf"), subMin=1, cpRate=-1,
             guardInput=1e-10, guardOutput=1e-10, guardDC=0,
             insertion=0, noisefigure=0, powerRF=power, powerLO=-3.56)
    return p


# ----------------------------------------------------------------------------- tab 1: principle

def spectrum_demo(N, M, seed):
    r = np.random.default_rng(seed)
    x = (0.3 + 0.7 * r.random(N)) * np.exp(2j * np.pi * r.random(N))
    W = (0.3 + 0.7 * r.random((N, M))) * np.exp(2j * np.pi * r.random((N, M)))
    K = N * M
    X = np.zeros(K, complex)
    X[::M] = x
    Wf = W.reshape(-1)[::-1].copy()
    # two spectra -> time-domain waveforms -> pointwise product (the mixer) -> spectrum
    C = np.fft.fft(np.fft.ifft(X) * np.fft.ifft(Wf)) * K
    t = np.arange(M)
    return dict(N=N, M=M, K=K, x=x, W=W, X=X, Wf=Wf, C=C, yA=C[K - 1 - t], yD=x @ W)


class PrincipleTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("page")
        self.seed, self.t = 11, 0

        self.n_spin = spin(2, 8, 4)
        self.m_spin = spin(2, 6, 3)
        self.t_spin = spin(0, 2, 0)
        reroll = QtWidgets.QPushButton("换一组随机数")
        for w in (self.n_spin, self.m_spin):
            w.valueChanged.connect(self.recompute)
        self.t_spin.valueChanged.connect(self.select)
        reroll.clicked.connect(self.reroll)

        ctl = QtWidgets.QHBoxLayout()
        for lab, w in (("输入长度 N", self.n_spin), ("输出长度 M", self.m_spin), ("查看输出 y", self.t_spin)):
            ctl.addWidget(QtWidgets.QLabel(lab))
            ctl.addWidget(w)
            ctl.addSpacing(14)
        ctl.addWidget(reroll)
        ctl.addStretch(1)

        self.plot = Plot(9, 5)
        self.plot.mpl_connect("button_press_event", self.on_click)
        self.explain = QtWidgets.QLabel()
        self.explain.setWordWrap(True)
        self.explain.setObjectName("metric")
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["输出", "直接计算 x·W[:, t]", "混频后的频点读数", "误差"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFont(mono())

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(lead("混频器把两路信号在时间上逐点相乘，等于两个频谱做卷积。输入 x 放在间隔为 M 的稀疏子载波上，"
                           "权重 W 按行展平再倒序铺满 N·M 个子载波。卷积结果的最后 M 个频点就是 y = xW。"
                           "下面真的做了 IDFT、逐点相乘和 DFT；点击图中读出窗口里的 y 可以切换查看。"))
        lay.addLayout(ctl)
        lay.addWidget(self.plot, 1)
        lay.addWidget(self.explain)
        lay.addWidget(self.table)
        self.recompute()

    def reroll(self):
        self.seed = (self.seed * 48271 + 7) % 2147483647
        self.recompute()

    def recompute(self):
        self.d = spectrum_demo(self.n_spin.value(), self.m_spin.value(), self.seed)
        self.t_spin.blockSignals(True)
        self.t_spin.setMaximum(self.d["M"] - 1)
        self.t = min(self.t, self.d["M"] - 1)
        self.t_spin.setValue(self.t)
        self.t_spin.blockSignals(False)
        self.redraw()

    def select(self, t):
        self.t = t
        self.redraw()

    def on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None or ev.inaxes is not getattr(self, "ax3", None):
            return
        K, M = self.d["K"], self.d["M"]
        j = int(round(ev.xdata))
        if K - M <= j <= K - 1:
            self.t_spin.setValue(K - 1 - j)

    def redraw(self):
        d, t = self.d, self.t
        N, M, K = d["N"], d["M"], d["K"]
        fig = self.plot.fig
        fig.clear()
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
        self.ax3 = ax3
        idx = np.arange(K)
        for ax in (ax1, ax2, ax3):
            ax.set_ylim(-0.42, 1.25)
            ax.set_yticks([])
            ax.axhline(0, color=C_RULE, lw=1)
            ax.spines["left"].set_visible(False)
        # x comb
        comb = idx[::M]
        stems(ax1, comb, np.abs(d["X"][comb]) / np.abs(d["X"]).max(), C_X)
        for n, i in enumerate(comb):
            ax1.text(i, -0.2, f"x{n}".translate(SUB), color=C_X, ha="center", va="center", fontsize=9)
        ax1.set_ylabel("输入 x\n稀疏梳状", color=C_X, rotation=0, ha="right", va="center", fontsize=10)
        # flattened, reversed W
        hl = {K - 1 - t - n * M for n in range(N)}
        hw = np.abs(d["Wf"]) / np.abs(d["Wf"]).max()
        on = np.array([i in hl for i in idx])
        stems(ax2, idx[~on], hw[~on], C_W, alpha=0.25)
        stems(ax2, idx[on], hw[on], C_W, ms=6)
        for g in range(N):
            ax2.plot([g * M - 0.35, g * M + M - 0.65], [-0.1, -0.1], color=C_W, lw=1, alpha=0.7)
            ax2.text(g * M + (M - 1) / 2, -0.27, f"W 第 {N - 1 - g} 行", color=C_W, ha="center", va="center", fontsize=8.5)
        ax2.set_ylabel("权重 W\n展平后倒序", color=C_W, rotation=0, ha="right", va="center", fontsize=10)
        # convolution
        hc = np.abs(d["C"]) / np.abs(d["C"]).max()
        ax3.axvspan(K - M - 0.5, K - 0.5, color=C_Y, alpha=0.08, lw=0)
        stems(ax3, idx[:K - M], hc[:K - M], C_MUTED, alpha=0.6, lw=1.6, ms=4)
        for j in range(K - M, K):
            tt = K - 1 - j
            sel = tt == t
            stems(ax3, [j], [hc[j]], C_Y, alpha=1 if sel else 0.5, ms=7 if sel else 5)
            ax3.text(j, -0.2, f"y{tt}".translate(SUB), color=C_Y, ha="center", va="center",
                     fontsize=10 if sel else 9, fontweight="bold" if sel else "normal")
        ax3.text((K - M + K - 1) / 2, 1.16, "读出窗口", color=C_Y, ha="center", fontsize=9)
        if K - M >= 3:
            ax3.text((K - M - 1) / 2, 1.16, "交叉项，接收端滤掉", color=C_MUTED, ha="center", fontsize=9)
        ax3.set_ylabel("混频输出\n频谱卷积", color=C_Y, rotation=0, ha="right", va="center", fontsize=10)
        ax3.set_xlabel("子载波序号")
        ax3.set_xlim(-0.8, K - 0.2)
        self.plot.draw_idle()

        self.explain.setText(
            f"<b style='color:{C_Y}'>y{t}</b> 落在第 {K - 1 - t} 号频点。卷积把每个梳齿 x<sub>n</sub> 和每组权重里"
            f"从左数第 {M - t} 个对齐相乘，这 {N} 个高亮的权重正好是 W 的第 {t} 列，"
            f"乘积之和就是 x 与这一列的内积。共占用 N·M = {K} 个子载波，等于这次矩阵乘的乘加次数。")
        self.table.setRowCount(M)
        self.table.setFixedHeight(self.table.horizontalHeader().height() + 30 * M + 4)
        for r in range(M):
            err = abs(d["yA"][r] - d["yD"][r])
            vals = [f"y{r}".translate(SUB), fmt_c(d["yD"][r]), fmt_c(d["yA"][r]), f"{err:.1e}"]
            self.table.setRowHeight(r, 30)
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if r == t:
                    it.setBackground(QtGui.QColor("#f6e3ec"))
                self.table.setItem(r, c, it)


# ----------------------------------------------------------------------------- tab 2: one layer

def run_layer(N, M, T, power, enc, progress=None):
    rng = np.random.default_rng()
    p = ws.get_param(low=True)
    p.update(transMode="easy", userNum=1, powerRF=power, encMode=enc)
    x = rng.random((T, N)) * np.exp(2j * np.pi * rng.random((T, N)))
    W = rng.random((T, N, M)) * np.exp(2j * np.pi * rng.random((T, N, M)))
    yd = np.einsum("tn,tnm->tm", x, W)
    t0 = time.time()
    out, snr, wave_time = ws.layer_fc(x, W, "auto", p, rng)
    ya = out[0, 0]
    ratio = yd / ya
    ph = np.angle(ratio / ratio.mean(axis=1, keepdims=True))
    return dict(yd=yd, ya=ya, snr=snr, wave_time=wave_time, elapsed=time.time() - t0, phase=ph,
                pearson=float(ws.calc_pearson(np.abs(yd), np.abs(ya)).mean()),
                rmse=float(ws.calc_rmse(np.abs(yd), np.abs(ya)).mean()),
                N=N, M=M, T=T, power=power, enc=enc)


class LayerTab(QtWidgets.QWidget):
    def __init__(self, pool):
        super().__init__()
        self.setObjectName("page")
        self.pool, self.job = pool, None

        self.n_spin = spin(10, 200, 100, 10)
        self.m_spin = spin(10, 200, 100, 10)
        self.t_spin = spin(1, 20, 5)
        self.enc = QtWidgets.QComboBox()
        self.enc.addItem("time（TestFC 默认）", "time")
        self.enc.addItem("freq", "freq")
        self.power = slider(-100, 0, -37)
        self.power_lab = QtWidgets.QLabel()
        self.noiseless = QtWidgets.QCheckBox("关闭噪声")
        self.power.valueChanged.connect(self.update_power_label)
        self.noiseless.toggled.connect(self.update_power_label)
        self.run_btn = QtWidgets.QPushButton("运行仿真")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.start)
        preset = QtWidgets.QPushButton("恢复 TestFC 原始设置")
        preset.clicked.connect(self.preset)

        form = QtWidgets.QFormLayout()
        form.addRow("输入长度 N", self.n_spin)
        form.addRow("输出长度 M", self.m_spin)
        form.addRow("矩阵组数", self.t_spin)
        form.addRow("编码模式", self.enc)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self.power, 1)
        prow.addWidget(self.power_lab)
        form.addRow("射频功率", prow)
        form.addRow("", self.noiseless)
        box = QtWidgets.QGroupBox("参数")
        bl = QtWidgets.QVBoxLayout(box)
        bl.addLayout(form)
        bl.addWidget(self.run_btn)
        bl.addWidget(preset)
        bl.addStretch(1)
        self.metrics = QtWidgets.QLabel("点击“运行仿真”。")
        self.metrics.setObjectName("metric")
        self.metrics.setWordWrap(True)
        bl.addWidget(self.metrics)
        box.setFixedWidth(330)

        self.plot = Plot(8, 4.5)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(lead("对应 Hybrid/TestFC.m：随机复数矩阵走一遍模拟射频链路，包括预处理、发射波形生成、"
                             "理想混频器加热噪声、接收解码和后处理，再和直接矩阵乘对比。"
                             "time 模式有约 2% 的固有误差，切到 freq 模式可以看到差别。"))
        right.addWidget(self.plot, 1)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(box)
        lay.addLayout(right, 1)
        self.update_power_label()
        self.empty_plot()

    def preset(self):
        self.n_spin.setValue(100)
        self.m_spin.setValue(100)
        self.t_spin.setValue(20)
        self.enc.setCurrentIndex(0)
        self.power.setValue(-37)
        self.noiseless.setChecked(False)

    def update_power_label(self):
        self.power.setEnabled(not self.noiseless.isChecked())
        self.power_lab.setText("无噪声" if self.noiseless.isChecked() else f"{self.power.value()} dBm")
        self.power_lab.setMinimumWidth(64)

    def args(self):
        power = 200.0 if self.noiseless.isChecked() else float(self.power.value())
        return self.n_spin.value(), self.m_spin.value(), self.t_spin.value(), power, self.enc.currentData()

    def start(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("仿真中…")
        self.job = Job(run_layer, *self.args())
        self.job.signals.done.connect(self.show_result)
        self.job.signals.failed.connect(self.fail)
        self.pool.start(self.job)

    def run_sync(self):
        self.show_result(run_layer(*self.args()))

    def fail(self, msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("运行仿真")
        self.metrics.setText(f"<span style='color:#b42318'>仿真失败：{msg}</span>")

    def empty_plot(self):
        self.plot.fig.clear()
        ax = self.plot.fig.add_subplot(111)
        ax.text(0.5, 0.5, "运行后这里显示：模拟输出与数字结果的散点图、相位误差分布", ha="center", va="center", color=C_MUTED)
        ax.axis("off")
        self.plot.draw_idle()

    def show_result(self, r):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("运行仿真")
        fig = self.plot.fig
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.2, 1]))
        a = np.abs(r["yd"]) / np.abs(r["yd"]).mean(axis=1, keepdims=True)
        b = np.abs(r["ya"]) / np.abs(r["ya"]).mean(axis=1, keepdims=True)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax1.plot([lo, hi], [lo, hi], color=C_RULE, lw=1.2, zorder=0)
        ax1.scatter(a, b, s=6, alpha=0.45, color=C_Y, lw=0)
        ax1.set_xlabel("数字计算 |xW|（归一化）")
        ax1.set_ylabel("模拟输出（归一化）")
        ax1.set_title(f"皮尔逊相关 {r['pearson']:.4f}", fontsize=11)
        ax2.hist(np.degrees(r["phase"]).ravel(), bins=40, color=C_X, alpha=0.8)
        ax2.set_xlabel("相位误差（度）")
        ax2.set_yticks([])
        ax2.spines["left"].set_visible(False)
        ax2.set_title(f"相位误差标准差 {np.degrees(np.std(r['phase'])):.1f}°", fontsize=11)
        self.plot.draw_idle()
        snr = "无噪声" if r["power"] > 100 else f"{r['snr']:.1f} dB"
        self.metrics.setText(
            f"<b>{r['T']} 组 {r['N']}×{r['M']}，{r['enc']} 模式</b><br>"
            f"信噪比 {snr}<br>皮尔逊相关 {r['pearson']:.4f}<br>RMSE {r['rmse']:.4f}<br>"
            f"波形时长 {r['wave_time'] * 1e3:.2f} ms<br>仿真耗时 {r['elapsed']:.1f} s")


# ----------------------------------------------------------------------------- tab 3: MNIST

def run_mnist_one(x, fc_list, power, mode, enc, progress=None):
    t0 = time.time()
    out, wave_time = ws.analog_inference(x[None], fc_list, "zadoff", "auto", mnist_param(power, mode, enc),
                                         np.random.default_rng())
    return dict(kind="one", ya=out[0, 0], wave_time=wave_time, elapsed=time.time() - t0, power=power)


def run_mnist_batch(X, Y, pred_dig, fc_list, power, mode, enc, progress=None):
    n, step, correct, agree = len(Y), 25, 0, 0
    rng = np.random.default_rng()
    t0 = time.time()
    for s in range(0, n, step):
        out, _ = ws.analog_inference(X[s:s + step], fc_list, "zadoff", "auto", mnist_param(power, mode, enc), rng)
        pred = np.argmax(out[0], axis=1)
        correct += int(np.sum(pred == Y[s:s + step]))
        agree += int(np.sum(pred == pred_dig[s:s + step]))
        if progress:
            progress(min(s + step, n), n)
    return dict(kind="batch", acc=correct / n, agree=agree / n, n=n, power=power, elapsed=time.time() - t0,
                acc_dig=float(np.mean(pred_dig == Y)))


class MnistTab(QtWidgets.QWidget):
    def __init__(self, pool, model, curve, demo):
        super().__init__()
        self.setObjectName("page")
        self.pool, self.job = pool, None
        self.fc_list, self.mac_num = model
        self.curve = curve
        self.X, self.Y = demo
        self.pred_dig = np.array([np.argmax(ws.digital_inference(x, self.fc_list, "zadoff")) for x in self.X])
        self.batch_points = []
        self.last = None
        self.pending = False
        # per-sample waveform duration, from the batch-of-100 runs in the curve file
        self.t_sample = curve["rows"][0]["wave_time_s"] / 100 if curve else 3.04e-4

        self.idx = spin(0, len(self.Y) - 1, 0)
        rnd = QtWidgets.QPushButton("随机一张")
        rnd.clicked.connect(lambda: self.idx.setValue(int(np.random.default_rng().integers(len(self.Y)))))
        self.idx.valueChanged.connect(self.on_sample)
        self.img = Plot(2.6, 2.6)
        self.truth = QtWidgets.QLabel()
        self.truth.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.power = slider(-110, -70, -76)
        self.power_lab = QtWidgets.QLabel()
        self.power_lab.setMinimumWidth(200)
        self.power.valueChanged.connect(self.on_power_move)
        self.power.sliderReleased.connect(self.start_one)
        self.power.valueChanged.connect(lambda _: None if self.power.isSliderDown() else self.debounce.start())
        self.debounce = QtCore.QTimer(singleShot=True, interval=250)
        self.debounce.timeout.connect(self.start_one)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("fast：收发两端都有噪声（main_dataset 默认）", "fast")
        self.mode.addItem("easy：理想混频器加接收噪声", "easy")
        self.enc = QtWidgets.QComboBox()
        self.enc.addItem("time（默认）", "time")
        self.enc.addItem("freq", "freq")
        self.mode.currentIndexChanged.connect(lambda _: self.start_one())
        self.enc.currentIndexChanged.connect(lambda _: self.start_one())
        self.run_btn = QtWidgets.QPushButton("再推理一次")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.start_one)
        self.batch_n = spin(25, len(self.Y), 100, 25)
        self.batch_btn = QtWidgets.QPushButton("批量评估")
        self.batch_btn.clicked.connect(self.start_batch)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setMaximumHeight(6)
        self.bar.hide()
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("metric")
        self.status.setWordWrap(True)

        # left column: sample
        sbox = QtWidgets.QGroupBox("测试样本")
        sl = QtWidgets.QVBoxLayout(sbox)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("编号"))
        row.addWidget(self.idx, 1)
        row.addWidget(rnd)
        sl.addLayout(row)
        sl.addWidget(self.img, 1)
        sl.addWidget(self.truth)
        sbox.setFixedWidth(270)

        # controls
        cbox = QtWidgets.QGroupBox("模拟射频链路")
        form = QtWidgets.QFormLayout(cbox)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self.power, 1)
        prow.addWidget(self.power_lab)
        form.addRow("射频输入功率", prow)
        form.addRow("信道模型", self.mode)
        form.addRow("编码模式", self.enc)
        brow = QtWidgets.QHBoxLayout()
        brow.addWidget(self.run_btn)
        brow.addSpacing(16)
        brow.addWidget(QtWidgets.QLabel("用前"))
        brow.addWidget(self.batch_n)
        brow.addWidget(QtWidgets.QLabel("张图"))
        brow.addWidget(self.batch_btn)
        brow.addStretch(1)
        form.addRow("", brow)
        form.addRow("", self.bar)

        self.bars = Plot(5, 3)
        self.curve_plot = Plot(5, 3)
        plots = QtWidgets.QHBoxLayout()
        plots.addWidget(self.bars, 1)
        plots.addWidget(self.curve_plot, 1)

        mid = QtWidgets.QVBoxLayout()
        mid.addWidget(lead("训练好的复数网络（196→64→32→10）逐层走模拟射频链路，参数沿用 Simulation/main_dataset.m。"
                           "拖动射频功率，功率越低、每次乘加的能耗越小，噪声也越大。右图是全部 10000 张测试图的预先仿真结果。"))
        mid.addWidget(cbox)
        mid.addLayout(plots, 1)
        mid.addWidget(self.status)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(sbox)
        lay.addLayout(mid, 1)

        self.on_power_move()
        self.draw_sample()
        self.draw_curve()
        self.draw_bars(None)

    # -- helpers
    def emac(self, power):
        return 10 ** (power / 10) / 1000 * self.t_sample / self.mac_num

    def on_power_move(self):
        p = self.power.value()
        self.power_lab.setText(f"{p} dBm · E_MAC {self.emac(p):.1e} J")
        self.draw_curve()

    def on_sample(self):
        self.draw_sample()
        self.start_one()

    def busy(self, on):
        for w in (self.run_btn, self.batch_btn):
            w.setEnabled(not on)

    # -- jobs
    def start_one(self):
        if self.job is not None:
            self.pending = True
            return
        self.pending = False
        self.busy(True)
        self.status.setText("推理中…")
        x = self.X[self.idx.value()]
        self.job = Job(run_mnist_one, x, self.fc_list, float(self.power.value()),
                       self.mode.currentData(), self.enc.currentData())
        self.job.signals.done.connect(self.on_done)
        self.job.signals.failed.connect(self.on_fail)
        self.pool.start(self.job)

    def start_batch(self):
        if self.job is not None:
            return
        self.busy(True)
        n = self.batch_n.value()
        self.bar.setRange(0, n)
        self.bar.setValue(0)
        self.bar.show()
        self.status.setText(f"批量评估 {n} 张图…")
        self.job = Job(run_mnist_batch, self.X[:n], self.Y[:n], self.pred_dig[:n], self.fc_list,
                       float(self.power.value()), self.mode.currentData(), self.enc.currentData())
        self.job.signals.progress.connect(lambda a, b: self.bar.setValue(a))
        self.job.signals.done.connect(self.on_done)
        self.job.signals.failed.connect(self.on_fail)
        self.pool.start(self.job)

    def run_sync(self):
        self.on_done(run_mnist_one(self.X[self.idx.value()], self.fc_list, float(self.power.value()),
                                   self.mode.currentData(), self.enc.currentData()))

    def on_fail(self, msg):
        self.job = None
        self.busy(False)
        self.bar.hide()
        self.status.setText(f"<span style='color:#b42318'>仿真失败：{msg}</span>")

    def on_done(self, r):
        self.job = None
        self.busy(False)
        self.bar.hide()
        if r["kind"] == "one":
            self.last = r
            self.draw_bars(r)
        else:
            self.batch_points.append((r["power"], r["acc"]))
            self.draw_curve()
            self.status.setText(
                f"<b>批量评估 {r['n']} 张，{r['power']:.0f} dBm</b>：模拟准确率 {r['acc'] * 100:.1f}%，"
                f"数字准确率 {r['acc_dig'] * 100:.1f}%，两者预测一致 {r['agree'] * 100:.1f}%，耗时 {r['elapsed']:.1f} s。"
                f"结果已作为星号画在右图上。")
        if self.pending:
            self.start_one()

    # -- drawing
    def draw_sample(self):
        i = self.idx.value()
        fig = self.img.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(np.abs(self.X[i]).reshape(14, 14), cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        self.img.draw_idle()
        self.truth.setText(f"<span style='font-size:15px'>真值 <b>{self.Y[i]}</b> · 数字预测 <b>{self.pred_dig[i]}</b></span>"
                           f"<br><span style='color:{C_MUTED}'>输入已乘 Zadoff-Chu 相位，图中是模值</span>")

    def draw_bars(self, r):
        i = self.idx.value()
        fig = self.bars.fig
        fig.clear()
        ax = fig.add_subplot(111)
        yd = ws.digital_inference(self.X[i], self.fc_list, "zadoff")
        k = np.arange(10)
        ax.bar(k - 0.2, yd / yd.max(), 0.38, color="#b8c2cc", label="数字")
        title = f"数字预测 {np.argmax(yd)}"
        if r is not None:
            ya = r["ya"]
            ax.bar(k + 0.2, ya / ya.max(), 0.38, color=C_Y, label="模拟")
            pa = int(np.argmax(ya))
            ok = "✓" if pa == self.Y[i] else "✗"
            title = f"数字预测 {np.argmax(yd)} · 模拟预测 {pa} {ok}"
            self.status.setText(
                f"单张推理：{r['power']:.0f} dBm，E_MAC {self.emac(r['power']):.1e} J，"
                f"波形时长 {r['wave_time'] * 1e3:.2f} ms，仿真耗时 {r['elapsed']:.2f} s。每次运行的噪声都是新抽的。")
        ax.set_xticks(k)
        ax.set_xlabel("类别")
        ax.set_ylabel("输出模值（归一化）")
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        self.bars.draw_idle()

    def draw_curve(self):
        fig = self.curve_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        p = self.power.value()
        if self.curve:
            rows = sorted(self.curve["rows"], key=lambda r: r["powerRF_dBm"])
            ax.plot([r["powerRF_dBm"] for r in rows], [r["acc_analog"] * 100 for r in rows], "-",
                    color=C_Y, lw=1.8, label="模拟（10000 张）")
            ax.axhline(self.curve["acc_digital"] * 100, color=C_MUTED, ls="--", lw=1, label="数字")
            acc_here = np.interp(p, [r["powerRF_dBm"] for r in rows], [r["acc_analog"] for r in rows]) * 100
            ax.plot([p], [acc_here], "o", color=C_Y, ms=7)
            ax.annotate(f"{acc_here:.1f}%", (p, acc_here), textcoords="offset points", xytext=(8, -14),
                        color=C_Y, fontsize=10, fontweight="bold")
        for bp, ba in self.batch_points:
            ax.plot([bp], [ba * 100], "*", color=C_X, ms=11)
        if self.batch_points:
            ax.plot([], [], "*", color=C_X, ms=9, label="本机批量评估")
        ax.axvline(p, color=C_FG, lw=0.8, alpha=0.4)
        ax.set_xlim(-111, -69)
        ax.set_ylim(0, 102)
        ax.set_xlabel("射频输入功率（dBm）")
        ax.set_ylabel("准确率（%）")
        ax.set_title("功率与准确率", fontsize=11)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        self.curve_plot.draw_idle()


# ----------------------------------------------------------------------------- main window

def load_model():
    m = loadmat(MODEL_PATH)
    fc, k = [], 1
    while f"Matrix_{k}" in m:
        fc.append(m[f"Matrix_{k}"].astype(complex).T)
        k += 1
    return fc, sum(a.shape[0] * a.shape[1] for a in fc)


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WISE 射频计算演示")
        self.resize(1320, 840)
        self.pool = QtCore.QThreadPool.globalInstance()
        fc_list, mac_num = load_model()
        curve = json.load(open(CURVE_PATH)) if os.path.exists(CURVE_PATH) else None
        demo = np.load(DEMO_PATH)
        self.tabs = QtWidgets.QTabWidget()
        self.principle = PrincipleTab()
        self.layer = LayerTab(self.pool)
        self.mnist = MnistTab(self.pool, (fc_list, mac_num), curve, (demo["x"], demo["y"]))
        self.tabs.addTab(self.principle, "原理演示")
        self.tabs.addTab(self.layer, "单层仿真")
        self.tabs.addTab(self.mnist, "MNIST 推理")
        self.setCentralWidget(self.tabs)
        shapes = " → ".join([str(fc_list[0].shape[1])] + [str(f.shape[0]) for f in fc_list])
        acc = f"，数字准确率 {curve['acc_digital'] * 100:.2f}%" if curve else ""
        self.statusBar().showMessage(f"模型 MNIST_size14_FC3：{shapes}，每次推理 {mac_num} 次乘加{acc}。"
                                     f"仿真代码是 functions-lab/WISE MATLAB 仿真的 numpy 移植版。")
        self.tabs.currentChanged.connect(self.on_tab)

    def on_tab(self, i):
        if self.tabs.widget(i) is self.mnist and self.mnist.last is None and self.mnist.job is None:
            self.mnist.start_one()


def screenshots(win, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    app = QtWidgets.QApplication.instance()
    for i, prep in enumerate([lambda: None, win.layer.run_sync, win.mnist.run_sync]):
        win.tabs.setCurrentIndex(i)
        app.processEvents()
        prep()
        for _ in range(5):
            app.processEvents()
        name = ["principle", "layer", "mnist"][i]
        win.grab().save(os.path.join(out_dir, f"{i + 1}_{name}.png"))
        print("saved", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", metavar="DIR", help="render the three pages offscreen, save PNGs, and exit")
    args = ap.parse_args()
    if args.screenshots and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = Main()
    win.show()
    if args.screenshots:
        win.mnist.debounce.stop()
        screenshots(win, args.screenshots)
        return 0
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
