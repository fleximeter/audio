# this code is from chat gpt

import numpy as np
import matplotlib.pyplot as plt
import librosa
import dev
from dev import mel, freq
import scipy.fft as fft
from aus import audiofile
import signal_plotter
from scipy.signal import ShortTimeFFT

####################################
# # This implementation shows that my Mel filterbanks are very similar but not quite identical. That is hopefully good enough for now. 
# FFT_SIZE = 2048
# SAMPLE_RATE = 44100
# NUM_MELS = 20
# mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=NUM_MELS, norm="slaney")
# rfreqs = fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
# fb = dev.MelFilterbank(0, SAMPLE_RATE // 2, NUM_MELS, rfreqs, True, True)

# mfl = librosa.core.mel_frequencies(128, fmin=0, fmax=SAMPLE_RATE//2)
# mel_center_freqs = np.linspace(mel(0), mel(SAMPLE_RATE//2), NUM_MELS+2)
# freq_center_freqs = np.zeros((NUM_MELS+2))
# for i in range(NUM_MELS+2):
#     freq_center_freqs[i] = freq(mel_center_freqs[i])
# mf = np.array(freq_center_freqs)

# print(mel_basis.shape)
# for i in range(NUM_MELS):
#     plt.plot(mel_basis[i])
# plt.show()

# plt.clf()
# for i in range(NUM_MELS):
#     plt.plot(fb.filterbank[i].triangle_filter)
# plt.show()

# VERIFICATIONS
EPSILON = 1e-8
# At this point, my mel conversion is ok
assert(np.abs(mel(0) - librosa.core.hz_to_mel(0)) < EPSILON)
assert(np.abs(mel(5) - librosa.core.hz_to_mel(5)) < EPSILON)
assert(np.abs(mel(100) - librosa.core.hz_to_mel(100)) < EPSILON)
assert(np.abs(mel(800) - librosa.core.hz_to_mel(800)) < EPSILON)
assert(np.abs(mel(1200) - librosa.core.hz_to_mel(1200)) < EPSILON)
assert(np.abs(mel(2000) - librosa.core.hz_to_mel(2000)) < EPSILON)
# My frequency conversion is also ok
assert(np.abs(freq(0) - librosa.core.mel_to_hz(0)) < EPSILON)
assert(np.abs(freq(3) - librosa.core.mel_to_hz(3)) < EPSILON)
assert(np.abs(freq(9) - librosa.core.mel_to_hz(9)) < EPSILON)
assert(np.abs(freq(15) - librosa.core.mel_to_hz(15)) < EPSILON)
assert(np.abs(freq(16) - librosa.core.mel_to_hz(16)) < EPSILON)
assert(np.abs(freq(21) - librosa.core.mel_to_hz(21)) < EPSILON)

# Compare spectrograms
FFT_SIZE = 1024
AUDIO = "D:/Recording/compress.wav"
WINDOW = np.hamming(FFT_SIZE)
af = audiofile.read(AUDIO)
rfft_freqs = fft.rfftfreq(FFT_SIZE, 1/af.sample_rate)
my_fb = dev.MelFilterbank(0, 22050, 128, rfft_freqs, False, True)
my_fb2 = np.vstack([item.triangle_filter for item in my_fb.filterbank])
lb_fb = librosa.filters.mel(sr=af.sample_rate, n_fft=FFT_SIZE, n_mels=128, fmin=0, fmax=22050)

# ok, maybe it's the way of computing spectrograms?
# this is the librosa way
my_S = librosa.stft(y=af.samples[0, :], n_fft=FFT_SIZE, hop_length=FFT_SIZE//2, win_length=FFT_SIZE, window="hann", center=True, pad_mode="constant")
my_powerspectrum = np.abs(my_S) ** 2

# this is the way I was doing it
stft_ = ShortTimeFFT(np.hamming(FFT_SIZE), FFT_SIZE // 2, af.sample_rate)
ispec = stft_.stft(af.samples[0, :])
pspec = np.square(np.abs(ispec))    

my_melspec = my_fb(pspec)
lb_melspec = np.einsum("...ft,mf->...mt", pspec, lb_fb, optimize=True)

signal_plotter.plot_spectrogram(my_melspec)
signal_plotter.plot_spectrogram(lb_melspec)
