"""Spectral characterisation of TAS1 RRP — wavelet + STFT.

Same shapes as the forecast repo. Used in §3.1 to motivate the
multi-scale state for the SDP.
"""

import numpy as np
import pywt


def cwt_morlet(prices, dt_minutes=30, n_scales=48,
               freq_low_hz=1 / (14 * 24 * 3600),
               freq_high_hz=1 / (15 * 60)):
    dt_seconds = dt_minutes * 60
    central_freq = pywt.central_frequency("cmor1.5-1.0")
    period_low = 1 / freq_high_hz
    period_high = 1 / freq_low_hz
    periods_seconds = np.geomspace(period_low, period_high, n_scales)
    scales = central_freq * periods_seconds / dt_seconds
    coeffs, _ = pywt.cwt(prices - prices.mean(), scales, "cmor1.5-1.0",
                         sampling_period=dt_seconds)
    return coeffs, periods_seconds / 3600


def stft(prices, dt_minutes=30, window_periods=336, hop=48):
    n = len(prices)
    win = np.hanning(window_periods)
    starts = np.arange(0, n - window_periods + 1, hop)
    spec = np.empty((len(starts), window_periods // 2 + 1))
    for k, s in enumerate(starts):
        chunk = (prices[s:s + window_periods] - prices[s:s + window_periods].mean()) * win
        F = np.fft.rfft(chunk)
        spec[k] = np.abs(F) ** 2
    freqs = np.fft.rfftfreq(window_periods, d=dt_minutes * 60)
    with np.errstate(divide="ignore"):
        periods_hours = np.where(freqs > 0, 1 / (freqs * 3600), np.nan)
    return spec, starts, periods_hours
