import librosa
import aus_analyzer
from aus import audiofile
import numpy as np
import scipy.fft as fft
import signal_plotter

FFT_SIZE = 2048
AUDIO = "D:/Recording/compress.wav"
WINDOW = np.hamming(FFT_SIZE)
af = audiofile.read(AUDIO)
rfft_freqs = fft.rfftfreq(FFT_SIZE, 1/af.sample_rate)

mspec = aus_analyzer.analyze_rstft("D:/Recording/compress.wav", 2048, 8)

signal_plotter.plot_spectrogram(mspec["mel_spectrogram"])
