import librosa
import aus_analyzer
from aus import audiofile
import numpy as np
import scipy.fft as fft
import mymel
from scipy.signal import ShortTimeFFT
import signal_plotter

FFT_SIZE = 2048
AUDIO = "D:/Recording/compress.wav"
WINDOW = np.hamming(FFT_SIZE)
af = audiofile.read(AUDIO)
rfft_freqs = fft.rfftfreq(FFT_SIZE, 1/af.sample_rate)

mspec = aus_analyzer.analyze_rstft("D:/Recording/compress.wav", 2048, 40, 20, 4)

#signal_plotter.plot_spectrogram(np.log10(np.square(mspec["magnitude_spectrogram"])))
signal_plotter.plot_spectrogram(mspec["mfccs"])
# print(mspec["mfccs"][18][:20])

NUM_MELS = 40
FMIN = 0
FMAX = 44100//2
stft_ = ShortTimeFFT(np.hamming(FFT_SIZE), FFT_SIZE // 2, af.sample_rate)
ispec = stft_.stft(af.samples[0, :])
pspec = np.square(np.abs(ispec))
f = mymel.MelFilterbank(FMIN, FMAX, NUM_MELS, rfft_freqs)
mel_specgram = f(pspec)
mel_specgram2 = librosa.feature.melspectrogram(sr=af.sample_rate, S=pspec, n_fft=FFT_SIZE, hop_length=FFT_SIZE//2, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX, norm="slaney")

# mfccs_me = mymel.mfcc(librosa.core.power_to_db(librosa.feature.melspectrogram(y=af.samples[0, :], sr=af.sample_rate, norm="slaney")))
mfccs_me = mymel.mfcc(mel_specgram)
mfccs_librosa = librosa.feature.mfcc(y=af.samples[0, :])
# signal_plotter.plot_spectrogram(mfccs_me[:20, ...])
signal_plotter.plot_spectrogram(mfccs_librosa)
