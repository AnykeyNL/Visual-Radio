"""A stand-in for SoapySDR with one fake Airspy, for tests/test_airspy_proc.py: samples
count up (real part) and carry the tuned frequency / 1e9 (imaginary part). Closing
the device aborts, like the real driver crashing."""
import os, numpy as np
SOAPY_SDR_RX=0; SOAPY_SDR_CF32="CF32"
class R:
    def __init__(s,n): s.ret=n
class Device:
    _n = 0
    @staticmethod
    def enumerate(a): return [{"driver": "airspy", "label": "Fake Airspy", "serial": "abc"}]
    def __init__(s,a): s.f = 0; s.k = 0
    def listSampleRates(s,*a): return [3e6, 6e6]
    def setSampleRate(s,*a): s.rate = a[-1]
    def getSampleRate(s,*a): return s.rate
    def hasGainMode(s,*a): return True
    def setGainMode(s,*a): pass
    def setGain(s,*a): s.g = a[-1]
    def getGain(s,*a): return s.g
    def getStreamMTU(s,st): return 65536
    def setFrequency(s,*a): s.f = a[-1]
    def setupStream(s,*a): return 1
    def activateStream(s,*a): pass
    def readStream(s,st,b,n,timeoutUs=0):
        # sample value = running counter + frequency/1e9 in imag
        b[0][:n] = (np.arange(s.k, s.k+n) % 1000).astype(np.float32) + 1j * np.float32(s.f / 1e9)
        s.k += n
        import time; time.sleep(n / s.rate)
        return R(n)
    def deactivateStream(s,st): pass
    def closeStream(s,st): pass
    def close(s): os.abort()
    @staticmethod
    def unmake(d): os.abort()
