"""IQ sample sources: a real Airspy Mini (via SoapySDR) and a simulator.

Every source offers the same small interface, so the scanner does not care
where the samples come from:

    src.sample_rate          -> float, samples per second
    src.tune(freq_hz)        -> retune the centre frequency
    src.read(n)              -> numpy complex64 array of n IQ samples
    src.close()
"""

from __future__ import annotations

import numpy as np


class DeviceError(RuntimeError):
    """Raised when the Airspy Mini cannot be found or opened."""


# --------------------------------------------------------------------------
# Real hardware
# --------------------------------------------------------------------------

SOAPY_HELP = """\
SoapySDR's Python module could not be imported.

On Windows the easiest fix is radioconda (see README.md, "Windows setup"):
  1. Install radioconda, open the "radioconda Prompt".
  2. Run this script from that prompt:  python visual_radio.py

On Raspberry Pi OS / Debian:
  sudo apt install python3-soapysdr soapysdr-module-airspy

Tip: run with --simulate to test the scanner without any hardware."""


def _import_soapy():
    try:
        import SoapySDR  # noqa: F401
        return SoapySDR
    except ImportError as exc:
        raise DeviceError(SOAPY_HELP) from exc


def list_airspy_devices() -> list[dict]:
    """Return the Airspy devices SoapySDR can see (empty list if none)."""
    SoapySDR = _import_soapy()
    found = SoapySDR.Device.enumerate("driver=airspy")
    # SoapySDRKwargs -> plain dict of str (works across SoapySDR versions).
    return [{str(k): str(d[k]) for k in d.keys()} for d in found]


class AirspySource:
    """Airspy Mini through SoapySDR + the SoapyAirspy module."""

    # The Airspy Mini supports 3 and 6 MSPS (10 MSPS only with special firmware).
    DEFAULT_RATE = 6e6

    def __init__(self, sample_rate: float = DEFAULT_RATE, gain: float = 18.0,
                 serial: str | None = None, settle_s: float = 0.03):
        SoapySDR = _import_soapy()
        self._soapy = SoapySDR

        devices = list_airspy_devices()
        if not devices:
            raise DeviceError(
                "No Airspy Mini found.\n"
                "  - Is it plugged in? Try another USB port or cable.\n"
                "  - Windows: the Airspy needs the WinUSB driver. Install it with\n"
                "    Zadig (see README.md) if SDR# or SDR++ cannot see it either.\n"
                "  - Linux / Pi: check 'lsusb' shows 1d50:60a1 and that the udev\n"
                "    rules are installed (or try once with sudo).\n"
                "  - Check 'SoapySDRUtil --find' lists the device.\n"
                "Tip: run with --simulate to test without hardware.")
        if serial:
            devices = [d for d in devices if serial.lower() in d.get("serial", "").lower()]
            if not devices:
                raise DeviceError(f"No Airspy with serial containing '{serial}' found.")

        self.info = devices[0]
        # Open with a plain "driver=...,serial=..." string: the most portable
        # form across SoapySDR versions and their Python bindings.
        open_args = "driver=airspy"
        if self.info.get("serial"):
            open_args += f",serial={self.info['serial']}"
        try:
            self.dev = SoapySDR.Device(open_args)
        except Exception as exc:  # SoapySDR raises a plain RuntimeError
            raise DeviceError(
                f"Airspy found but could not be opened ({open_args}): {exc}\n"
                "Is another program (SDR#, SDR++, GQRX) using it? Close it and retry."
            ) from exc

        rx = SoapySDR.SOAPY_SDR_RX
        rates = self.dev.listSampleRates(rx, 0)
        if rates and not any(abs(r - sample_rate) < 1 for r in rates):
            raise DeviceError(
                f"Sample rate {sample_rate/1e6:g} MSPS not supported. "
                f"This device offers: {', '.join(f'{r/1e6:g}' for r in rates)} MSPS")
        self.dev.setSampleRate(rx, 0, sample_rate)
        self.sample_rate = self.dev.getSampleRate(rx, 0)

        # Manual gain gives repeatable measurements (AGC would change between chunks).
        if self.dev.hasGainMode(rx, 0):
            self.dev.setGainMode(rx, 0, False)
        self.dev.setGain(rx, 0, gain)
        self.gain = self.dev.getGain(rx, 0)

        self.settle_s = settle_s
        self.stream = self.dev.setupStream(rx, SoapySDR.SOAPY_SDR_CF32)
        self.dev.activateStream(self.stream)
        self._mtu = max(4096, self.dev.getStreamMTU(self.stream))
        self._buf = np.empty(self._mtu, dtype=np.complex64)

    def describe(self) -> str:
        label = self.info.get("label") or self.info.get("product") or "Airspy"
        serial = self.info.get("serial", "?")
        return (f"{label} (serial {serial}), {self.sample_rate/1e6:g} MSPS, "
                f"gain {self.gain:g} dB")

    def tune(self, freq_hz: float) -> None:
        self.dev.setFrequency(self._soapy.SOAPY_SDR_RX, 0, freq_hz)
        # Throw away samples captured while the tuner's PLL settles.
        self.read(int(self.sample_rate * self.settle_s))

    def read(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.complex64)
        got = 0
        failures = 0
        while got < n:
            want = min(self._mtu, n - got)
            sr = self.dev.readStream(self.stream, [self._buf], want, timeoutUs=1_000_000)
            if sr.ret > 0:
                out[got:got + sr.ret] = self._buf[:sr.ret]
                got += sr.ret
                failures = 0
            else:
                # Negative codes: timeout (-1) or overflow (-4). Overflows are
                # harmless here; persistent timeouts mean the device is gone.
                failures += 1
                if failures > 20:
                    raise DeviceError(f"Airspy stopped delivering samples (code {sr.ret}).")
        return out

    def close(self) -> None:
        """Stop streaming. The device itself stays open until the program ends: on some
        PCs SoapySDR's Airspy driver crashes (access violation) when it closes the device.
        To hand the Airspy to another program, use airspy_proc.AirspyProcess instead."""
        dev = getattr(self, "dev", None)
        if dev is None:
            return
        try:
            dev.deactivateStream(self.stream)
            dev.closeStream(self.stream)
        except Exception:
            pass
        _KEEP.append(dev)                    # never garbage-collected (= closed) while running


_KEEP: list = []


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

# Made-up test band: (frequency MHz, dB above the noise floor[, RDS name[, radio text]]).
DEFAULT_SIM_STATIONS = [
    (88.3, 25, "SIM 88"), (89.6, 32, "TESTFM"), (91.8, 18), (92.6, 40, "RADIO 1", "Now playing: test tone"),
    (94.8, 12), (96.8, 35, "DEMO"), (98.9, 28), (100.7, 45, "BIG FM", "The strongest test station"),
    (101.2, 9), (103.8, 30, "LOCAL"), (104.4, 22), (106.5, 15, "WEAK"), (107.2, 6),
]

MOD_RATE = 250e3  # the simulator builds each station's modulation at this rate


class _SimStation:
    """One fake FM transmitter: mono audio + 19 kHz pilot (+ RDS on 57 kHz)."""

    def __init__(self, freq_mhz, snr_db, name=None, text=None, pi=0x8000, rng=None):
        from . import rds
        self.freq_hz = freq_mhz * 1e6
        self.snr_db = snr_db
        self.name = name
        rng = rng or np.random.default_rng()
        self.tones = rng.uniform(300, 6000, 3)
        self.pilot_phase = rng.uniform(0, 2 * np.pi)
        self.phase0 = rng.uniform(0, 2 * np.pi)
        self.state = None  # (grid index, modulation phase) for phase continuity
        self.tx_bits = None
        if name:
            groups = rds.ps_groups(pi, name)
            if text:
                for i, rt in enumerate(rds.rt_groups(pi, text)):
                    groups.insert(min(len(groups), (i + 1) * 2), rt)
            bits = [b for g in groups for b in rds.encode_group(g)]
            bits = np.array(bits * max(1, int(30 * rds.BIT_RATE) // len(bits)), dtype=np.uint8)
            self.tx_bits = np.bitwise_xor.accumulate(bits)  # differential encoding

    def mpx(self, k: np.ndarray) -> np.ndarray:
        """FM baseband (+/-1 = +/-75 kHz deviation) at grid indices k (t = k / MOD_RATE)."""
        t = k / MOD_RATE
        m = sum(0.25 * np.sin(2 * np.pi * f * t) for f in self.tones)
        pilot = 2 * np.pi * 19e3 * t + self.pilot_phase
        m = m + 0.09 * np.cos(pilot)
        if self.tx_bits is not None:
            from .rds import BIT_RATE
            pos = t * BIT_RATE
            idx = pos.astype(np.int64)
            a = np.where(self.tx_bits[idx % len(self.tx_bits)] == 1, 1.0, -1.0)
            sym = np.where(pos - idx >= 0.5, -a, a)          # bi-phase symbol
            m = m + 0.05 * sym * np.cos(3 * pilot)
        return m

    def phase(self, t: np.ndarray) -> np.ndarray:
        """Modulation phase (radians) at times t, continuous across calls."""
        k_first = int(np.floor(t[0] * MOD_RATE))
        k_last = int(np.ceil(t[-1] * MOD_RATE)) + 1
        if self.state is None or self.state[0] > k_first:
            self.state = (k_first, 0.0)
        k0, ph0 = self.state
        k = np.arange(k0, k_last + 1)
        steps = 2 * np.pi * 75e3 / MOD_RATE * self.mpx(k[1:])
        ph = np.concatenate([[ph0], ph0 + np.cumsum(steps)])
        keep = int(np.floor(t[-1] * MOD_RATE)) - k0
        self.state = (k0 + keep, ph[keep])
        return np.interp(t * MOD_RATE, k, ph)


class SimulatedSource:
    """Generates IQ samples of a fake FM band so the scanner can run without hardware.

    Each station is a real FM signal (75 kHz deviation) with mono audio tones,
    a 19 kHz stereo pilot and, if it has a name, RDS on 57 kHz carrying that
    name and an optional radio text. It is placed at its true offset from the
    tuned frequency, on top of white noise. Stations outside the current
    window are not present, exactly like on a real receiver.
    """

    def __init__(self, sample_rate: float = 6e6, stations=None,
                 noise_dbfs: float = -60.0, seed: int | None = 1):
        self.sample_rate = float(sample_rate)
        self.rng = np.random.default_rng(seed)
        spec = stations if stations is not None else DEFAULT_SIM_STATIONS
        self.stations = [_SimStation(*s, pi=0x8201 + i, rng=self.rng) for i, s in enumerate(spec)]
        self.noise_dbfs = noise_dbfs
        self.center = 98e6
        self.clock = 0  # total samples produced, so time runs on across reads

    def describe(self) -> str:
        return (f"SIMULATED Airspy Mini, {self.sample_rate/1e6:g} MSPS, "
                f"{len(self.stations)} fake stations")

    def tune(self, freq_hz: float) -> None:
        self.center = float(freq_hz)

    def read(self, n: int) -> np.ndarray:
        fs = self.sample_rate
        t = (self.clock + np.arange(n)) / fs
        self.clock += n
        # Noise power per sample, spread over the whole sample bandwidth.
        noise_pow = 10 ** (self.noise_dbfs / 10)
        x = (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) * np.sqrt(noise_pow / 2)

        # "Above the noise floor" is measured in a ~120 kHz channel, so scale
        # the carrier power to the noise in that bandwidth.
        noise_in_channel = noise_pow * 120e3 / fs
        half = fs / 2
        for st in self.stations:
            offset = st.freq_hz - self.center
            if abs(offset) > half - 100e3:
                continue
            amp = np.sqrt(noise_in_channel * 10 ** (st.snr_db / 10))
            ph = 2 * np.pi * np.mod(offset * t, 1.0) + st.phase(t) + st.phase0
            x += amp * np.exp(1j * ph)
        return x.astype(np.complex64)

    def close(self) -> None:
        pass


def parse_sim_stations(text: str):
    """Parse '88.3:25:MYNAME,100.7:45' into [(88.3, 25.0, 'MYNAME'), (100.7, 45.0)]."""
    stations = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        entry = [float(fields[0]), float(fields[1]) if len(fields) > 1 and fields[1] else 30.0]
        if len(fields) > 2 and fields[2]:
            entry.append(fields[2])
        stations.append(tuple(entry))
    return stations
