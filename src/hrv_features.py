# features_hrv.py
import numpy as np
from scipy import signal, interpolate

def compute_time_domain(rr_ms: np.ndarray) -> dict:
    rr = rr_ms.astype(float)
    if len(rr) < 2:
        return {}
    diff = np.diff(rr)
    feats = {}
    feats["meanNN"] = np.mean(rr)
    feats["sdnn"]   = np.std(rr, ddof=1)
    feats["rmssd"]  = np.sqrt(np.mean(diff**2))
    nn50 = np.sum(np.abs(diff) > 50.0)
    feats["pnn50"]  = nn50 / max(1, len(rr)-1)
    # Poincaré
    sd1 = np.sqrt(np.var((rr[1:] - rr[:-1]) / np.sqrt(2), ddof=1))
    sd2 = np.sqrt(np.var((rr[1:] + rr[:-1]) / np.sqrt(2), ddof=1))
    feats["sd1"] = sd1
    feats["sd2"] = sd2
    feats["sd2_sd1"] = (sd2 / sd1) if sd1 > 1e-6 else 0.0
    return feats

def _rr_to_series(rr_ms: np.ndarray, fs=4):
    """将不等间隔RR转换为等间隔心率时间序列(Hz=fs)。"""
    # 以毫秒为单位的RR -> 累积时间轴(s)
    t = np.cumsum(rr_ms)/1000.0
    t = t - t[0]
    hr_inst = 60000.0 / rr_ms  # 即时心率(bpm)
    # 线性插值到等采样
    t_uniform = np.arange(0, t[-1], 1.0/fs)
    f = interpolate.interp1d(t, hr_inst, kind="linear", fill_value="extrapolate")
    hr_u = f(t_uniform)
    return t_uniform, hr_u

def compute_freq_domain(rr_ms: np.ndarray, fs_resample=4) -> dict:
    if len(rr_ms) < 4:
        return {}
    t, hr_u = _rr_to_series(rr_ms, fs=fs_resample)
    # 去均值
    x = hr_u - np.mean(hr_u)
    f, pxx = signal.welch(x, fs=fs_resample, nperseg=min(256, len(x)))
    # 频带功率
    def band_power(f, pxx, lo, hi):
        idx = (f>=lo) & (f<=hi)
        return np.trapz(pxx[idx], f[idx]) if np.any(idx) else 0.0
    lf = band_power(f, pxx, 0.04, 0.15)
    hf = band_power(f, pxx, 0.15, 0.40)
    tp = band_power(f, pxx, 0.003, 0.40)
    feats = {
        "lf": lf, "hf": hf,
        "lf_hf": (lf / hf) if hf > 1e-9 else np.inf,
        "lfn": lf / tp if tp > 0 else 0.0,
        "hfn": hf / tp if tp > 0 else 0.0,
    }
    return feats

def compute_hrv_features(rr_ms: np.ndarray) -> dict:
    """汇总HRV特征：时间域+频域（与研究一致，如LF/HF、SD2/SD1等）。"""
    if len(rr_ms) < 2:
        return {}
    feats = {}
    feats.update(compute_time_domain(rr_ms))
    feats.update(compute_freq_domain(rr_ms))
    return feats
