"""
File: dev.py

This file is for experimenting.
"""

import aus.audiofile as audiofile
import signal_plotter
import synthesis
import scipy.fft as fft
import numpy as np

sin = synthesis.sine(440, 0, 88200, 44100)
shift = synthesis.freqshift(sin, 100, 44100)
spec = fft.rfft(shift)
sig = synthesis.complex_sinusoid(440, 100, 44100)
# print(sig.dtype)
signal_plotter.plot_spectrum(spec, 44100, (0, 1000))