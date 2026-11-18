"""Sweep the FM band in receiver-sized chunks and find stations.

How it works
------------
1. The FM band (87.5-108 MHz) is wider than the Airspy Mini can see at once
   (6 MSPS -> about 6 MHz). We step the centre frequency so the usable middle
   part of each window (default 80 %) tiles the band without gaps.
2. At each step we read IQ samples and compute an averaged power spectrum
   (windowed FFT, many frames averaged = Welch's method).
3. The usable bins from all steps are stitched into one band-wide spectrum.
4. Noise floor = a low percentile of all bins (robust even in a busy band).
5. We slide a 120 kHz measuring window across the band in 10 kHz steps. Each
   peak that is `threshold_db` above the noise in the same bandwidth is a
   station, snapped to the nearest 100 kHz channel. Using peaks (not every
   channel above the threshold) stops one strong signal, which spills into
   the neighbouring channels, from being reported several times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BAND_START_HZ = 87.5e6
BAND_STOP_HZ = 108.0e6
GRID_HZ = 100e3
CHANNEL_HALF_WIDTH_HZ = 60e3


@dataclass
class Station:
    freq_hz: float
    power_dbfs: float   # power in the channel, relative to ADC full scale
    snr_db: float       # how far above the noise floor (same bandwidth)

    @property
    def freq_mhz(self) -> float:
        return self.freq_hz / 1e6


@dataclass
class ScanResult:
    freqs_hz: np.ndarray      # stitched spectrum: bin frequencies
    psd: np.ndarray           # stitched spectrum: linear power per bin
    noise_per_bin: float      # linear noise power per bin
    stations: list[Station]


def plan_centers(start_hz: float, stop_hz: float, sample_rate: float,
                 usable_fraction: float) -> list[float]:
    """Centre frequencies whose usable windows cover [start, stop]."""
    usable = sample_rate * usable_fraction
    # Pad by one channel so the edge stations are measured fully.
    lo = start_hz - 2 * CHANNEL_HALF_WIDTH_HZ
    hi = stop_hz + 2 * CHANNEL_HALF_WIDTH_HZ
    n = int(np.ceil((hi - lo) / usable))
    # Spread the steps evenly so overlap is shared instead of all at the end.
    first = lo + usable / 2
    last = hi - usable / 2
    if n <= 1:
        return [(lo + hi) / 2]
    return list(np.linspace(first, last, n))


def power_spectrum(iq: np.ndarray, nfft: int) -> np.ndarray:
    """Averaged, fftshifted power spectrum, scaled so that the sum of all
    bins equals the mean signal power (|x|^2) of the input."""
    frames = len(iq) // nfft
    if frames < 1:
        raise ValueError(f"need at least {nfft} samples, got {len(iq)}")
    window = np.hanning(nfft).astype(np.float32)
    x = iq[:frames * nfft].reshape(frames, nfft) * window
    spec = np.abs(np.fft.fft(x, axis=1)) ** 2
    psd = spec.mean(axis=0) / (nfft * np.sum(window ** 2))
    return np.fft.fftshift(psd)


def sweep(source, start_hz=BAND_START_HZ, stop_hz=BAND_STOP_HZ, nfft=8192,
          averages=64, usable_fraction=0.8, progress=None):
    """Tune across the band and return the stitched (freqs, psd) arrays."""
    fs = source.sample_rate
    centers = plan_centers(start_hz, stop_hz, fs, usable_fraction)
    bin_offsets = (np.arange(nfft) - nfft // 2) * fs / nfft
    usable = np.abs(bin_offsets) <= fs * usable_fraction / 2

    all_f, all_p = [], []
    for i, fc in enumerate(centers):
        if progress:
            progress(i, len(centers), fc)
        source.tune(fc)
        iq = source.read(nfft * averages)
        psd = power_spectrum(iq, nfft)
        f = fc + bin_offsets
        # Where two windows overlap, each keeps the half closest to its own
        # centre, so no frequency is counted twice.
        keep = usable.copy()
        if i > 0:
            keep &= f >= (centers[i - 1] + fc) / 2
        if i < len(centers) - 1:
            keep &= f < (fc + centers[i + 1]) / 2
        all_f.append(f[keep])
        all_p.append(psd[keep])

    return np.concatenate(all_f), np.concatenate(all_p)


def _window_sums(freqs, values, centers, half_width):
    """Sum of `values` with |freq - centre| <= half_width, for every centre."""
    cs = np.concatenate([[0.0], np.cumsum(values)])
    lo = np.searchsorted(freqs, centers - half_width, side="left")
    hi = np.searchsorted(freqs, centers + half_width, side="right")
    return cs[hi] - cs[lo], hi - lo


def find_stations(freqs, psd, start_hz=BAND_START_HZ, stop_hz=BAND_STOP_HZ,
                  threshold_db=10.0, noise_percentile=25.0, fine_step_hz=10e3):
    """Detect stations on the 100 kHz grid. Returns (stations, noise_per_bin).

    The channel power is first computed on a fine 10 kHz grid. Stations are the
    local maxima of that curve (within +/-50 kHz), snapped to the nearest
    100 kHz channel. Working on the fine grid means a weak station 200 kHz
    from a strong one is still found: the strong station's skirt only slopes
    down, it never forms a peak of its own.
    """
    in_band = (freqs >= start_hz - CHANNEL_HALF_WIDTH_HZ) & (freqs <= stop_hz + CHANNEL_HALF_WIDTH_HZ)
    noise_per_bin = float(np.percentile(psd[in_band], noise_percentile))

    fine = np.arange(start_hz - GRID_HZ, stop_hz + GRID_HZ + 1, fine_step_hz)
    power, nbins = _window_sums(freqs, psd, fine, CHANNEL_HALF_WIDTH_HZ)
    snr = 10 * np.log10(np.maximum(power, 1e-30) / (noise_per_bin * np.maximum(nbins, 1)))

    reach = int(round(GRID_HZ / 2 / fine_step_hz))
    best = {}  # grid channel -> Station
    for k in range(len(fine)):
        if snr[k] < threshold_db:
            continue
        lo, hi = max(0, k - reach), min(len(fine), k + reach + 1)
        if power[k] < power[lo:hi].max() or np.argmax(power[lo:hi]) != k - lo:
            continue  # not the (first) highest point in its neighbourhood
        ch = round(fine[k] / GRID_HZ) * GRID_HZ
        if not (start_hz - 1 <= ch <= stop_hz + 1) or abs(fine[k] - ch) > 0.4 * GRID_HZ:
            continue
        j = int(round((ch - fine[0]) / fine_step_hz))  # measure at the grid point
        st = Station(float(ch), float(10 * np.log10(power[j])), float(snr[j]))
        if ch not in best or st.snr_db > best[ch].snr_db:
            best[ch] = st
    stations = [best[ch] for ch in sorted(best) if best[ch].snr_db >= threshold_db]
    return stations, noise_per_bin


def scan(source, threshold_db=10.0, nfft=8192, averages=64,
         start_hz=BAND_START_HZ, stop_hz=BAND_STOP_HZ, progress=None) -> ScanResult:
    freqs, psd = sweep(source, start_hz, stop_hz, nfft, averages, progress=progress)
    stations, noise = find_stations(freqs, psd, start_hz, stop_hz, threshold_db)
    return ScanResult(freqs, psd, noise, stations)
