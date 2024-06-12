"""
File: sample_shifting.py

An experimentation with shifting samples and STFT frames in time
"""

import audiopython.audiofile as audiofile
import audiopython.operations as operations
import numpy as np
import os
import scipy.signal as signal

if __name__ == "__main__":
    DIR = "C:\\Users\\jeffr\\Recording\\reaper"
    FILE = os.path.join(DIR, "voice1.wav")
    OUT = os.path.join(DIR, "voice1.1.wav")
    FFT_SIZE = 2048
    audio = audiofile.read(FILE)
    lpf = signal.butter(2, 3000, "low", output="sos", fs=audio.sample_rate)
    # new_samples1 = signal.sosfilt(lpf, exchanger(audio.samples, 8))
    STFT = signal.ShortTimeFFT(signal.windows.hann(FFT_SIZE), FFT_SIZE // 2, audio.sample_rate)
    stft_data = STFT.stft(audio.samples)
    stft_data = operations.stochastic_exchanger(stft_data, 8)
    out1 = audiofile.AudioFile.copy_header(audio)
    out1.samples = np.array(STFT.istft(stft_data))
    out1.samples = operations.leak_dc_bias_filter(out1.samples)
    out1.samples = operations.adjust_level(out1.samples, -12)
    audiofile.write_with_pedalboard(out1, OUT)
