from scipy.signal import ShortTimeFFT
import numpy as np
import soundfile as sf

audio, sr = sf.read("D:\\Recording\\Samples\\Miscellaneous\\cello.wav")
audio = np.sum(audio, axis=1)
print(audio.shape)
FFT_SIZE = 2048
STFT = ShortTimeFFT(np.hanning(FFT_SIZE), FFT_SIZE//2, sr)
X = STFT.stft(audio)
M = np.abs(X)
P = np.angle(X)
PV = np.zeros(P.shape, dtype=np.float64)
print(P.shape)
for i in range(1, P.shape[-1]):
    PV[:, i] = P[:, i] - P[:, i-1]
NM = np.zeros((P.shape[0], P.shape[1] * 2))
NP = np.zeros((P.shape[0], P.shape[1] * 2))
NP[:, 0] = PV[:, 0]
NP[:, 1] = PV[:, 0] * 2
for i in range(2, NP.shape[-1]):
    NP[:, i] = NP[:, i-1] + PV[:, i]
NX = M * np.exp(1j * NP)
na = STFT.istft(NX)
sf.write("D:\\Recording\\test.wav", na, sr)