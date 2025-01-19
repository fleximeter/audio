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
import torchaudio, torchaudio.transforms
import torch

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


def plot_pyplot_spectrogram(file: str):
    audio = audiofile.read(file)
    plt.specgram(audio.samples[0], 2048, Fs=audio.sample_rate)
    plt.show()


def plot_spectrogram(file: str):
    audio = audiofile.read(file)
    a = aus_analyzer.analyze_rstft(file, 2048, 8)
    times = [(1024 * (i+1)) / audio.sample_rate for i in range(a["magnitude_spectrogram"].shape[0])]
    a["log_spectrogram"] = 20 * np.log10(np.square(np.swapaxes(a["magnitude_spectrogram"], 0, 1))) 
    plt.pcolormesh(times, a["rfftfreqs"], a["log_spectrogram"], shading="gouraud")
    plt.colorbar(label="Power (dB)")
    plt.title("Spectrogram")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_mel_spectrogram(file: str):
    audio = audiofile.read(file)
    a = aus_analyzer.analyze_rstft(file, 2048, 8)
    times = [(1024 * (i+1)) / audio.sample_rate for i in range(a["magnitude_spectrogram"].shape[0])]
    a["log_mel_spectrogram"] = 20 * np.log10(np.swapaxes(a["mel_spectrogram"], 0, 1)) 
    plt.pcolormesh(times, a["mel_scale"], a["log_mel_spectrogram"], shading="gouraud")
    plt.colorbar(label="Power (dB)")
    plt.title("Mel Spectrogram")
    plt.ylabel("Mels")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_torchaudio_mel_spectrogram(file: str):
    audio, sr = torchaudio.load(file, normalize=True)
    t = torchaudio.transforms.MelSpectrogram(sr, 2048, f_min=80, f_max=8000, n_mels=26)
    ms = t(audio)
    ms = ms.numpy()
    ms = ms[0]
    # a = aus_analyzer.analyze_rstft(file, 2048, 8)
    # m = librosa.feature.melspectrogram(S=np.swapaxes(a["magnitude_spectrogram"], 0, 1), n_fft=2048, hop_length=1024, fmin=20, fmax=8000, n_mels=26)
    # times = [(1024 * (i+1)) / audio.sample_rate for i in range(m.shape[-1])]
    plt.pcolormesh(ms, shading="gouraud")
    plt.colorbar(label="Power (dB)")
    plt.title("Mel Spectrogram")
    plt.ylabel("Mels")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_mfccs(file: str):
    audio = audiofile.read(file)
    a = aus_analyzer.analyze_rstft(file, 2048, 8)
    times = [(1024 * (i+1)) / audio.sample_rate for i in range(a["magnitude_spectrogram"].shape[0])]
    a["mfccs"] = np.swapaxes(a["mfccs"], 0, 1)
    plt.pcolormesh(times, a["mel_scale"][:6], a["mfccs"][:6], shading="gouraud")
    plt.colorbar(label="Power (dB)")
    plt.title("MFCCs")
    plt.ylabel("Mels")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# audio = audiofile.read("D:\\Recording\\compress.wav")
# auto = autocorrelation(audio.samples[0, 44100:44100+2048])
# auto2 = librosa.autocorrelate(audio.samples[0, 44100:44100+2048])
# plt.plot([i for i in range(0, 2048)], auto2)
# plt.show()
plot_torchaudio_mel_spectrogram("D:\\Recording\\compress.wav")
