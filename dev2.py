# this code is from chat gpt

import numpy as np
import matplotlib.pyplot as plt
import librosa
import dev
from dev import mel, freq
import scipy.fft as fft

FFT_SIZE = 2048
SAMPLE_RATE = 44100
NUM_MELS = 20
mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=NUM_MELS, norm="slaney")
rfreqs = fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
fb = dev.MelFilterbank(0, SAMPLE_RATE // 2, NUM_MELS, rfreqs, False, True)

mfl = librosa.core.mel_frequencies(128, fmin=0, fmax=SAMPLE_RATE//2)
mel_center_freqs = np.linspace(mel(0), mel(SAMPLE_RATE//2), NUM_MELS+2)
freq_center_freqs = np.zeros((NUM_MELS+2))
for i in range(NUM_MELS+2):
    freq_center_freqs[i] = freq(mel_center_freqs[i])
mf = np.array(freq_center_freqs)

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
