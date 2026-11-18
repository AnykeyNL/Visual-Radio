"""How hard is this PC working? CPU, RAM and GPU load for the web page.

- CPU and RAM come from psutil (pip install psutil), or without it straight from
  Windows (GetSystemTimes / GlobalMemoryStatusEx) or Linux (/proc).
- The NVIDIA GPU comes from NVML (pip install nvidia-ml-py), or else from the
  nvidia-smi tool that comes with the driver.
Whatever is missing is simply left out; nothing here may stop the receiver.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time


def _gb(n):
    return round(n / 2**30, 2)


class _Psutil:
    def __init__(self):
        import psutil
        self.ps = psutil
        psutil.cpu_percent(None)                  # the first call only starts the clock
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.name = "psutil"

    def read(self):
        ps = self.ps
        out = {"cpu": round(ps.cpu_percent(None), 1), "cores": ps.cpu_count() or 1}
        vm = ps.virtual_memory()
        out["ram"], out["ram_used"], out["ram_total"] = round(vm.percent, 1), _gb(vm.used), _gb(vm.total)
        out["proc"] = round(self.proc.cpu_percent(None) / out["cores"], 1)
        return out


class _OsCpu:
    """CPU and RAM without psutil: Windows API through ctypes, or /proc on Linux."""

    def __init__(self):
        self.cores = os.cpu_count() or 1
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            self.k32 = ctypes.windll.kernel32
            self.ct, self.wt = ctypes, wintypes
            P = ctypes.POINTER(wintypes.FILETIME)
            self.k32.GetSystemTimes.argtypes = [P, P, P]
            self.k32.GetSystemTimes.restype = wintypes.BOOL
            self.k32.GetCurrentProcess.restype = wintypes.HANDLE
            self.k32.GetProcessTimes.argtypes = [wintypes.HANDLE, P, P, P, P]
            self.k32.GetProcessTimes.restype = wintypes.BOOL
            self.me = self.k32.GetCurrentProcess()     # pseudo handle, needs no closing
        elif not os.path.exists("/proc/stat"):
            raise RuntimeError("no CPU counters")
        self.name = "Windows" if sys.platform == "win32" else "/proc"
        self.last = self._times()

    def _times(self):
        """-> (idle, busy+idle of all cores, this process) in seconds, plus wall clock."""
        if sys.platform == "win32":
            ct, ft = self.ct, self.wt.FILETIME
            v = lambda f: ((f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e7
            idle, kern, user = ft(), ft(), ft()
            if not self.k32.GetSystemTimes(ct.byref(idle), ct.byref(kern), ct.byref(user)):
                raise OSError("GetSystemTimes failed")
            c, e, pk, pu = ft(), ft(), ft(), ft()
            self.k32.GetProcessTimes(self.me, ct.byref(c), ct.byref(e), ct.byref(pk), ct.byref(pu))
            return v(idle), v(kern) + v(user), v(pk) + v(pu), time.monotonic()   # kernel time includes idle
        with open("/proc/stat") as f:
            vals = [float(x) for x in f.readline().split()[1:]]
        hz = os.sysconf("SC_CLK_TCK")
        t = os.times()
        return (vals[3] + (vals[4] if len(vals) > 4 else 0)) / hz, sum(vals[:8]) / hz, t.user + t.system, time.monotonic()

    def _ram(self):
        if sys.platform == "win32":
            ct = self.ct

            class MEM(ct.Structure):
                _fields_ = [("dwLength", ct.c_ulong), ("dwMemoryLoad", ct.c_ulong),
                            ("ullTotalPhys", ct.c_ulonglong), ("ullAvailPhys", ct.c_ulonglong),
                            ("ullTotalPageFile", ct.c_ulonglong), ("ullAvailPageFile", ct.c_ulonglong),
                            ("ullTotalVirtual", ct.c_ulonglong), ("ullAvailVirtual", ct.c_ulonglong),
                            ("ullAvailExtendedVirtual", ct.c_ulonglong)]
            m = MEM()
            m.dwLength = ct.sizeof(MEM)
            if not self.k32.GlobalMemoryStatusEx(ct.byref(m)):
                return None
            return m.ullTotalPhys, m.ullTotalPhys - m.ullAvailPhys
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"], info["MemTotal"] - info.get("MemAvailable", info.get("MemFree", 0))

    def read(self):
        now = self._times()
        (i0, a0, p0, w0), (i1, a1, p1, w1) = self.last, now
        self.last = now
        out = {"cores": self.cores}
        if a1 - a0 > 0:
            out["cpu"] = round(max(0.0, min(100.0, 100 * (1 - (i1 - i0) / (a1 - a0)))), 1)
        if w1 - w0 > 0:
            out["proc"] = round(max(0.0, min(100.0, 100 * (p1 - p0) / ((w1 - w0) * self.cores))), 1)
        ram = self._ram()
        if ram and ram[0]:
            total, used = ram
            out["ram"], out["ram_used"], out["ram_total"] = round(100 * used / total, 1), _gb(used), _gb(total)
        return out


class _Nvml:
    def __init__(self, index=0):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(self.h)
        self.name = name.decode() if isinstance(name, bytes) else str(name)

    def read(self):
        nv, h = self.nv, self.h
        out = {"name": self.name}
        try:
            out["util"] = int(nv.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            pass
        try:
            m = nv.nvmlDeviceGetMemoryInfo(h)
            out["mem_used"], out["mem_total"] = _gb(m.used), _gb(m.total)
        except Exception:
            pass
        try:
            out["temp"] = int(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            out["power"] = round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 1)
            out["power_max"] = round(nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000)
        except Exception:
            pass
        return out


class _NvidiaSmi:
    FIELDS = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"

    def __init__(self, index=0):
        self.exe = shutil.which("nvidia-smi")
        if not self.exe:
            raise RuntimeError("no nvidia-smi")
        self.index = index
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        self.name = self.read().get("name", "GPU")

    def read(self):
        r = subprocess.run([self.exe, f"--id={self.index}", f"--query-gpu={self.FIELDS}",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=5, creationflags=self.flags)
        return parse_smi(r.stdout)


def parse_smi(line):
    """One nvidia-smi csv line -> the same dict as NVML gives."""
    parts = [p.strip() for p in (line.strip().splitlines() or [""])[0].split(",")]
    if len(parts) < 7:
        return {}
    out = {"name": parts[0]}

    def num(s):
        try:
            return float(s)
        except ValueError:          # "[N/A]"
            return None
    util, used, total, temp, power, limit = (num(p) for p in parts[1:7])
    if util is not None:
        out["util"] = int(util)
    if used is not None and total:
        out["mem_used"], out["mem_total"] = round(used / 1024, 2), round(total / 1024, 2)
    if temp is not None:
        out["temp"] = int(temp)
    if power is not None:
        out["power"] = round(power, 1)
        if limit:
            out["power_max"] = round(limit)
    return out


class SystemLoad:
    """Samples the load every `interval` s in its own thread and hands each sample
    to on_sample(dict): {"cpu": %, "cores": n, "ram": %, "ram_used": GB, "ram_total": GB,
    "proc": % of the whole CPU used by this program, "gpu": {...} or None}."""

    def __init__(self, on_sample, interval=1.0, gpu_index=0, extra=None):
        self.on_sample = on_sample
        self.interval = interval
        self.extra = extra            # optional fn() -> dict merged into every sample
        self.stop = threading.Event()
        self.cpu = None
        for kind in (_Psutil, _OsCpu):
            try:
                self.cpu = kind()
                break
            except Exception:
                continue
        self.gpu = None
        for kind in (_Nvml, _NvidiaSmi):
            try:
                self.gpu = kind(gpu_index)
                break
            except Exception:
                continue
        self.gpu_every = 1 if isinstance(self.gpu, _Nvml) else 2     # nvidia-smi is a program start
        self.last_gpu = None
        self.n = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def description(self):
        parts = []
        parts.append(f"CPU ({self.cpu.name})" if self.cpu else "no CPU reading (pip install psutil)")
        parts.append(f"GPU {self.gpu.name}" if self.gpu else "no NVIDIA GPU found")
        return ", ".join(parts)

    def start(self):
        self.thread.start()
        return self

    def sample(self):
        out = {"t": round(time.time(), 1)}
        if self.cpu:
            try:
                out.update(self.cpu.read())
            except Exception as exc:
                if not getattr(self, "_cpu_error", None):
                    self._cpu_error = str(exc)
                    print(f"\r\x1b[KCPU load could not be read: {exc}")
        if self.gpu and self.n % self.gpu_every == 0:
            try:
                self.last_gpu = self.gpu.read() or self.last_gpu
            except Exception:
                pass
        self.n += 1
        out["gpu"] = self.last_gpu
        if self.extra:
            try:
                out.update(self.extra())
            except Exception:
                pass
        return out

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                self.on_sample(self.sample())
            except Exception:
                pass
