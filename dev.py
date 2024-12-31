"""
File: dev.py

This file is for experimenting.
"""

import aus.audiofile as audiofile
import aus.operations as operations
import aus.synthesis as synthesis
import pedalboard as pb
import datetime
import numpy as np
from scipy import fft
from matplotlib import pyplot as plt
import librosa
import aus_analyzer

def autocorrelation(audio: np.ndarray):
    """
    An implementation of the autocorrelation function using the FFT
    (Eyben, 44-45)
    """
    zplen = audio.size // 2
    newaudio = np.hstack((np.zeros((zplen)), audio, np.zeros((zplen))))
    spectrum = fft.fft(newaudio)
    conj = np.conj(spectrum)
    correlation = fft.ifft(spectrum * conj)
    return correlation


audio = audiofile.read("D:\\Recording\\compress.wav")
auto = autocorrelation(audio.samples[0, 44100:44100+2048])
auto2 = librosa.autocorrelate(audio.samples[0, 44100:44100+2048])
plt.plot([i for i in range(0, 2048)], auto2)
plt.show()