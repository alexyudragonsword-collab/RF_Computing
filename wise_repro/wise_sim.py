"""numpy port of the WISE analog-computing simulation (MATLAB, functions-lab/WISE @ 9e05926).

Function-by-function translation of what Hybrid/TestFC.m and
Hybrid/Simulation/main_dataset.m call:

    GetParam / GetParam_low                      -> get_param
    LayerFC (non-"full" path)                    -> layer_fc
    PreUniting / PreSlicing / PrePadding         -> pre_*
    LayerFC_generateTx                           -> generate_tx
    Tx2Rx_all + tx2rx_easy + tx2rx_fast          -> tx2rx_all / tx2rx_easy / tx2rx_fast
    LayerFC_analyzeRx                            -> analyze_rx
    PostPadding / PostSlicing / PostUniting      -> post_*
    AnalogInference / DigitalInference           -> analog_inference / digital_inference
    Activation, myZadoff, CalcPearson, CalcRMSE  -> same names in snake_case

MATLAB is 1-based and rounds halves away from zero; both are handled explicitly.
Deviations from the original are marked with "DEVIATION".
"""
import math

import numpy as np

K_BOLTZMANN = 1.380649e-23  # physconst("Boltzmann")


def mround(x):
    """MATLAB round(): halves go away from zero (Python's round() goes to even)."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


def iround(x):
    return int(mround(x))


# ----------------------------------------------------------------------------- params

def get_param(low=False):
    """GetParam.m (low=False) and GetParam_low.m (low=True)."""
    p = dict(
        transMode="exp", encMode="time", decMode="split-1",
        sampleRate=25e6 if low else 100e6,
        analog=6e9, impedance=50, boltzmann=K_BOLTZMANN, temperature=300,
        powerLO=-50, insertion=11.97,
        powerRF=-37, noisefigure=28,          # "Easy" values overwrite the "Fast" ones
        delay=0, calib=[], precode="weight", cpRate=-1, padRate=0.333,
        inputNorm=0.2, weightNorm=0.2, userNum=1, attenList=[0], atten=float("nan"),
        batchInOutProd=1e8, batchOutProd=1e5, autoEdge=125 if low else 500,
        carrierTx=[1.2e9, 0.915e9 if low else 0.9e9], convert="down",
        gainTx=[9, 19], gainRx=20,
    )
    # transMode is "exp" when GetParam runs, so the "exp" guard values apply
    p.update(subMax=1000000, subMin=1, guardDC=1e-5, guardInput=0.05, guardOutput=0.1)
    if p["convert"] == "up":
        p["carrierRx"] = p["carrierTx"][0] + p["carrierTx"][1]
    else:
        p["carrierRx"] = abs(p["carrierTx"][0] - p["carrierTx"][1])
    return p


# ----------------------------------------------------------------------------- helpers

def my_zadoff(n, r=29):
    c = n % 2
    k = np.arange(n)
    return np.exp(-1j * np.pi * r * k * (k + c) / n)


def activation(x, act):
    """Activation.m; x is (batch, feat)."""
    if act == "zadoff":
        return np.abs(x) * my_zadoff(x.shape[1])[None, :]
    if act == "relu":
        return np.maximum(x, 0)
    if act == "conv":
        return np.stack([np.convolve(row, row) for row in x])
    raise ValueError(act)


def calc_pearson(x_all, y_all):
    r = []
    for x, y in zip(x_all, y_all):
        xm, ym = x - x.mean(), y - y.mean()
        r.append(np.sum(xm * ym) / np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)))
    return np.array(r)


def calc_rmse(x_all, y_all):
    r = []
    for x, y in zip(x_all, y_all):
        r.append(np.sqrt(np.mean(np.abs(x / x.mean() - y / y.mean()) ** 2)))
    return np.array(r)


def dftmtx(n):
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n)


# ----------------------------------------------------------------------------- pre/post processing

def pre_uniting(inp, w, p):
    B, N, M = w.shape
    unite = int(math.ceil(math.sqrt(p["subMin"] / N / M)))
    if unite > 1:
        raise NotImplementedError("PreUniting with unite > 1 is not used by the ported scripts")
    return inp, w, dict(unite=unite)


def pre_slicing(inp, w, p):
    U = inp.shape[0]
    B, N, M = w.shape
    dec = p["decMode"]
    if dec == "inner":
        n_slice = M
    elif dec.startswith("split-"):
        n_slice = int(math.ceil(M / iround(float(dec[6:]))))
    else:
        n_slice = M if N > p["subMax"] else max(int(math.ceil(M / math.floor(p["subMax"] / N))), 1)
    Ms = int(math.ceil(M / n_slice))
    info = dict(slice=n_slice, B=B, M=M)
    if n_slice == 1:
        return inp, w, info
    in_s = np.zeros((U, B * n_slice, N), complex)
    w_s = np.zeros((B * n_slice, N, Ms), complex)
    end = 0
    for s in range(1, n_slice + 1):
        span = int(math.ceil(M / n_slice)) if s <= M % n_slice else M // n_slice
        start, end = end, end + span
        idx = np.arange(B) * n_slice + (s - 1)
        in_s[:, idx, :] = inp
        w_s[idx, :, :span] = w[:, :, start:end]
    return in_s, w_s, info


def pre_padding(inp, w, shrink, p):
    U = inp.shape[0]
    B, N0, M0 = w.shape
    g_dc = int(math.ceil(N0 * p["guardDC"]))
    middle = int(math.ceil(N0 / 2))
    g_in = int(math.ceil(N0 * p["guardInput"]))
    if shrink > 0:
        tmp = N0 + 6 * g_dc + 2 * g_in
        Np = int(math.ceil(tmp / shrink)) * shrink
        delta = iround((Np - tmp) / 2)
        left0 = g_dc + g_in + delta
        right0 = g_dc * 5 + g_in + delta + middle
    else:
        Np = N0 + 6 * g_dc + 2 * g_in
        left0 = g_dc + g_in
        right0 = g_dc * 5 + g_in + middle
    g_out = int(math.ceil(M0 * p["guardOutput"]))
    Mp = M0 + 2 * g_out
    in_p = np.zeros((U, B, Np), complex)
    in_p[:, :, left0:left0 + middle] = inp[:, :, :middle]
    in_p[:, :, right0:right0 + N0 - middle] = inp[:, :, middle:]
    w_p = np.zeros((B, Np, Mp), complex)
    w_p[:, left0:left0 + middle, g_out:g_out + M0] = w[:, :middle, :]
    w_p[:, right0:right0 + N0 - middle, g_out:g_out + M0] = w[:, middle:, :]
    return in_p, w_p, dict(out_slice=slice(g_out, g_out + M0), subOffset=g_dc * Mp)


def pre_processing(inp, w, shrink, p):
    i1, w1, unite = pre_uniting(inp, w, p)
    i2, w2, sl = pre_slicing(i1, w1, p)
    i3, w3, pad = pre_padding(i2, w2, shrink, p)
    return i3, w3, dict(unite=unite, slice=sl, pad=pad)


def post_processing(out, info):
    out = out[:, info["pad"]["out_slice"]]                      # PostPadding
    sl = info["slice"]                                          # PostSlicing
    n_slice, B, M = sl["slice"], sl["B"], sl["M"]
    res = np.zeros((B, M), complex)
    end = 0
    for s in range(1, n_slice + 1):
        span = int(math.ceil(M / n_slice)) if s <= M % n_slice else M // n_slice
        start, end = end, end + span
        res[:, start:end] = out[np.arange(B) * n_slice + (s - 1), :span]
    return res                                                  # PostUniting is identity for unite == 1


# ----------------------------------------------------------------------------- transmit side

def _add_cp(wave, cp_len, pad_len):
    """AddCP for a (rows, L) block; MATLAB's length() is the long dimension here."""
    rows, L = wave.shape
    if cp_len == 0:
        cp = wave
    else:
        mult = np.tile(wave, (1, int(math.ceil(cp_len / L / 2))))
        h = iround(cp_len / 2)
        cp = np.concatenate([mult[:, mult.shape[1] - h:], wave, mult[:, :cp_len - h]], axis=1)
    if pad_len == 0:
        return cp
    pl = pad_len // 2
    return np.concatenate([np.zeros((rows, pl)), cp, np.zeros((rows, pad_len - pl))], axis=1)


def _encode_wave(x, freq_offset, csi):
    return np.fft.ifft(np.roll(x / csi, freq_offset, axis=-1), axis=-1)


def generate_tx(inp, w, repeat, sub_offset, shape, p):
    """LayerFC_generateTx.m; inp (U, S, N), w (S, N, M). Vectorised over symbols."""
    S, N, M = shape
    U = inp.shape[0]
    if p["precode"] != "weight" or p["calib"] or p["delay"] != 0:
        raise NotImplementedError("channel-state precoding needs calibration files from real hardware")
    cp_rate = p["cpRate"] if p["cpRate"] >= 0 else 2 / M
    cp_len = N * iround(M * cp_rate)
    pad_len = N * iround(M * p["padRate"])
    batch_len = N * M + cp_len + pad_len
    in_norm = p["inputNorm"] * math.sqrt((batch_len - pad_len) / batch_len)
    w_norm = p["weightNorm"] * math.sqrt((batch_len - pad_len) / batch_len)
    wave_len = S * (N * M + cp_len)
    n_w = iround(S / repeat)

    w = w.copy()
    if p["encMode"] == "time":
        sing = inp
        f_off = iround(sub_offset / M) - int(math.ceil(N / 2))
        trans_enc = np.roll(np.conj(dftmtx(N)) / N, -f_off, axis=0) / M      # EncodeWaveTrans
        w[:n_w] = np.einsum("ij,sjk->sik", trans_enc, w[:n_w])
    else:
        f_off = iround(sub_offset / M) - int(math.ceil(N / 2))
        sing = _encode_wave(inp, f_off, np.ones(N)) / M
    if p["decMode"] == "time":
        trans_dec = np.roll(dftmtx(M), int(math.ceil(M / 2)), axis=1)        # DecodeWaveTrans
        w[:n_w] = np.einsum("sij,jk->sik", w[:n_w], trans_dec)
    if in_norm < 0:
        sing = -in_norm * sing / np.sqrt(np.mean(np.abs(sing) ** 2, axis=2, keepdims=True))
    wave_in = _add_cp(np.tile(sing.reshape(U * S, N), (1, M)), cp_len, pad_len).reshape(U, S * batch_len)
    if in_norm > 0:
        wave_in = in_norm * wave_in / np.sqrt(np.mean(np.abs(wave_in) ** 2, axis=1, keepdims=True))

    w_conv = w[:n_w].reshape(n_w, N * M)[:, ::-1]   # flip(reshape(permute(w,[1 3 2]),S,[]),2)
    sing_w = _encode_wave(w_conv, sub_offset - int(math.ceil(N / 2)) * M, np.ones(N * M))
    if w_norm < 0:
        sing_w = -w_norm * sing_w / np.sqrt(np.mean(np.abs(sing_w) ** 2, axis=1, keepdims=True))
    wave_w = _add_cp(sing_w, cp_len, pad_len).reshape(-1)
    if w_norm > 0:
        wave_w = w_norm * wave_w / np.sqrt(np.mean(np.abs(wave_w) ** 2))
    wave_w = np.tile(wave_w, repeat)

    if p["convert"] == "down":
        if p["carrierTx"][0] > p["carrierTx"][1]:
            wave_w = np.conj(wave_w)
        else:
            wave_in = np.conj(wave_in)
    return wave_in, wave_w, wave_len


# ----------------------------------------------------------------------------- channel models

def _upsampling(x, rate_low, rate_high):
    L = len(x)
    Lh = iround(L / rate_low * rate_high)
    spec = np.fft.fft(x) * rate_high / rate_low
    mid = (L + 1) // 2 if L % 2 == 1 else L // 2
    out = np.zeros(Lh, complex)
    out[:mid] = spec[:mid]
    out[Lh - L + mid:] = spec[mid:]
    return np.fft.ifft(out)


def _my_filter(x, f_low, f_high):
    L = len(x)
    ratio = f_low / f_high
    mask = np.zeros(L)
    mask[:int(math.ceil(L * ratio / 2))] = 1
    b = int(math.floor(L * ratio / 2))
    if b > 0:
        mask[L - b:] = 1
    return np.fft.ifft(np.fft.fft(x) * mask)


def _cnoise(rng, n, sigma):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(sigma / 2)


def tx2rx_easy(p1, p2, shrink, freq, p, rng):
    """Library/tx2rx_easy/tx2rx_easy.m: ideal mixer, brick-wall low-pass, thermal noise."""
    fs = p["sampleRate"]
    fs_rx = fs / shrink
    up = 4
    t = np.arange(1, up * len(p1) + 1) / up / fs
    w1, w2 = _upsampling(p1, fs, up * fs), _upsampling(p2, fs, up * fs)
    rot = np.exp(-1j * 2 * np.pi * freq * t)
    if p["convert"] == "up":
        rx = w1 * w2 * rot
    elif p["carrierTx"][0] > p["carrierTx"][1]:
        rx = w1 * np.conj(w2) * rot
    else:
        rx = np.conj(w1) * w2 * rot
    rx = _my_filter(rx, fs_rx, up * fs)
    power = p["powerRF"] - p["atten"]
    snr = 10 ** ((power - 30 - p["noisefigure"]) / 10) / (p["boltzmann"] * p["temperature"] * fs)
    sigma = np.mean(np.abs(rx) ** 2) / (1 + p["padRate"]) / snr
    rx = rx + _cnoise(rng, len(rx), sigma)
    return rx[up * shrink - 1::up * shrink], 10 * math.log10(snr)


def tx2rx_fast(p1, p2, shrink, freq, p, rng):
    """Library/tx2rx_fast/tx2rx_fast.m: noisy RF and LO inputs, mixer, receiver noise."""
    def add_noise(sig, noise, power, fs):
        snr = 10 ** ((power - 30) / 10) / (p["boltzmann"] * p["temperature"] * fs)
        sigma = np.mean(np.abs(sig) ** 2) / snr
        return sig + _cnoise(rng, len(sig), sigma), noise + _cnoise(rng, len(noise), sigma)

    power1, power2 = p["powerRF"] - p["atten"], p["powerLO"]
    fs = p["sampleRate"]
    fs_rx = fs / shrink
    up = 2
    pad_len = 100 * shrink
    pad_tx = np.zeros(pad_len, complex)
    t = np.arange(1, up * len(p1) + 1) / up / fs
    tp = np.arange(1, up * pad_len + 1) / up / fs
    w1, pad1 = add_noise(p1, pad_tx, power1, fs)
    w1, pad1 = _upsampling(w1, fs, up * fs), _upsampling(pad1, fs, up * fs)
    w2, pad2 = add_noise(p2, pad_tx, power2, fs)
    w2, pad2 = _upsampling(w2, fs, up * fs), _upsampling(pad2, fs, up * fs)
    if p["convert"] == "up":
        rx = w1 * w2 * np.exp(-1j * 2 * np.pi * freq * t)
    else:
        rx = w1 * np.conj(w2) * np.exp(-1j * 2 * np.pi * freq * t)
    pad_rx = pad1 * pad2 * np.exp(-1j * 2 * np.pi * freq * tp)
    power_rx = power1 - p["insertion"] - p["noisefigure"]
    rx, pad_rx = add_noise(rx, pad_rx, power_rx, up * fs)
    rx = _my_filter(rx, fs_rx, up * fs)[up * shrink - 1::up * shrink]
    pad_rx = _my_filter(pad_rx, fs_rx, up * fs)[up * shrink - 1::up * shrink]
    wp, pp = np.mean(np.abs(rx) ** 2), np.mean(np.abs(pad_rx) ** 2)
    if wp <= pp:
        snr = power1 - p["insertion"] - p["noisefigure"] - 10 * math.log10(
            p["boltzmann"] * p["temperature"] * fs) - 30
    else:
        snr = 10 * math.log10((wp - pp) / pp)
    return rx, snr


def tx2rx_all(wave_in, wave_w, shrink, freq_offset, p, rng):
    mode = p["transMode"]
    if wave_in.shape[0] != 1:
        # The original easy branch passes the whole multi-row matrix and assigns
        # outside its loop (Tx2Rx_all.m:21-23), so multi-user easy mode is broken there.
        raise NotImplementedError("multi-user transmission is not ported")
    if mode == "easy":
        out, snr = tx2rx_easy(wave_in[0], wave_w, shrink, freq_offset * p["sampleRate"], p, rng)
    elif mode == "fast":
        # DEVIATION: this branch is commented out in Tx2Rx_all.m, although
        # Simulation/main_dataset.m selects transMode = "fast". Re-enabled here.
        out, snr = tx2rx_fast(wave_in[0], wave_w, shrink, freq_offset * p["sampleRate"], p, rng)
    else:
        raise NotImplementedError(f"transMode {mode!r} needs USRP hardware")
    return np.tile(out, (p["userNum"], 1)), snr, 0.0


# ----------------------------------------------------------------------------- receive side

def analyze_rx(wave_out, span, shape, p):
    """LayerFC_analyzeRx.m."""
    S, N, M = shape
    cp_rate = p["cpRate"] if p["cpRate"] >= 0 else 2 / M
    cp_len = iround(M * cp_rate) * span
    pad_len = span * iround(M * p["padRate"])
    batch_len = M * span + cp_len + pad_len
    offset = int(math.floor((span - 1) / 2 * M))
    blk = wave_out[:S * batch_len].reshape(S, batch_len)
    if cp_len:
        h = iround(cp_len / 2)
        blk = blk[:, h:batch_len + h - cp_len]
    if pad_len:
        pl = pad_len // 2
        blk = blk[:, pl:blk.shape[1] - (pad_len - pl)]
    if p["decMode"] != "time":
        blk = np.roll(np.fft.fft(blk, axis=1), int(math.ceil(blk.shape[1] / 2)), axis=1)
    out = blk[:, offset:offset + M][:, ::-1]
    if N % 2 == 0 and M % 2 == 1:
        out = np.roll(out, 1, axis=1)
    return out


# ----------------------------------------------------------------------------- one layer

def layer_fc(inp, weight, method, p, rng):
    """LayerFC.m. Returns (out[atten, user, test, M], snr, wave_time)."""
    if inp.ndim == 3:
        U, T, N = inp.shape
        inp_l = inp
    else:
        T, N = inp.shape
        inp_l = np.tile(inp[None], (p["userNum"], 1, 1))
    if weight.ndim == 3:
        M, w_l, repeat = weight.shape[2], weight, 1
    elif weight.shape[0] == T and weight.shape[1] == N:
        # Original "Vector Inner Product" branch. Note: a square N x N weight with a
        # batch of exactly N samples also lands here, which is a bug in the original.
        M, w_l, repeat = 1, weight[:, :, None], 1
    else:
        M, w_l, repeat = weight.shape[1], np.tile(weight[None], (T, 1, 1)), T

    shrink = -1
    if method.startswith("down-"):
        shrink = iround(float(method[5:]))
    elif method == "auto" and p["transMode"] == "exp" and N > p["autoEdge"]:
        shrink = p["autoEdge"]
    if method == "full":
        raise NotImplementedError("method 'full' is not ported")

    in_n, w_n, info = pre_processing(inp_l, w_l, shrink, p)
    sub_offset = info["pad"]["subOffset"]
    S, Nn, Mn = w_n.shape
    wave_in, wave_w, wave_len = generate_tx(in_n, w_n, repeat, sub_offset, (S, Nn, Mn), p)
    if Nn % 2 == 1:
        sub_offset -= Mn // 2
    wave_time = wave_len / p["sampleRate"]

    U = inp_l.shape[0]
    atten_list = p["attenList"]
    out = np.full((len(atten_list), U, T, M), np.nan, complex)
    snr = None
    for a_i, atten in enumerate(atten_list):
        pa = dict(p, atten=atten)
        sh = Nn if shrink < 0 else shrink
        f_off = (2 * sub_offset - iround(Mn / 2)) / Nn / Mn
        wave_out, snr, _ = tx2rx_all(wave_in / math.sqrt(10 ** (atten / 10)), wave_w, sh, f_off, pa, rng)
        for u in range(U):
            o = analyze_rx(wave_out[u], iround(Nn / sh), (S, Nn, Mn), pa)
            out[a_i, u] = post_processing(o, info)
    return out, snr, wave_time


# ----------------------------------------------------------------------------- whole network

def digital_inference(x, fc_list, act):
    """DigitalInference.m with fc_list entries stored as (out, in), like FCList in MATLAB."""
    data = x
    for fc in fc_list[:-1]:
        data = activation((data @ fc.T)[None, :], act)[0]
    data = data @ fc_list[-1].T
    return data if act == "relu" else np.abs(data)


def analog_inference(inputs, fc_list, act, method, p, rng):
    """AnalogInference.m (fully connected path; its conv path is commented out upstream)."""
    U = p["userNum"]
    D = inputs.shape[0]
    data = np.tile(inputs.reshape(1, D, -1), (U, 1, 1))
    if p["transMode"] == "exp":
        b_io = p["batchInOutProd"]
        b_o = float("inf") if p["decMode"] == "freq" else p["batchOutProd"]
    else:
        b_io = b_o = -1
    wave_time = 0.0
    for i, fc_stored in enumerate(fc_list):
        fc = fc_stored.T                                   # (in, out)
        n_in, n_out = fc.shape
        if b_io < 0 or b_o < 0:
            batch = 100
        else:
            batch = max(min(math.floor(b_io / n_in / n_out), math.floor(b_o / n_out)), 1)
        tmp = np.zeros((U, D, n_out), complex)
        for b in range(int(math.ceil(D / batch))):
            s, e = b * batch, min((b + 1) * batch, D)
            o, _, wt = layer_fc(data[:, s:e, :], fc, method, p, rng)
            tmp[:, s:e, :] = o.reshape(U, e - s, n_out)
            wave_time += wt
        data = tmp
        if i < len(fc_list) - 1:
            data = activation(data.reshape(U * D, -1), act).reshape(U, D, -1)
    return np.abs(data), wave_time
