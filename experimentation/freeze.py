# File: freeze.py
# This file tests FFT freezing.

import numpy as np
import scipy.fft as fft
import scipy.signal as signal
import pedalboard as pb
from typing import Tuple

INPUT_AUDIO = "D:\\Recording\\Samples\\Iowa\\Cello.arco.mono.2444.1\\..."
OUTPUT_AUDIO = "D:\\Recording\\out_a.wav"
FFT_SIZE = 2048

def freeze(magnitude_spectrogram, phase_spectrogram, freeze_index, num_frames) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements a spectral freeze
    :param magnitude_spectrogram: The magnitude spectrogram
    :param phase_spectrogram: The phase spectrogram
    :param freeze_index: The index at which to freeze
    :param num_frames: The number of output frames to generate
    """
    freeze_frame = magnitude_spectrogram[:, freeze_index]
    phase_differences = phase_spectrogram[:, freeze_index + 1] - phase_spectrogram[:, freeze_index]
    running_phase = phase_spectrogram[:, freeze_index]
    new_mags = np.repeat(freeze_frame, num_frames)
    new_phases = [phase_spectrogram[:, freeze_index]]
    for _ in range(1, num_frames):
        new_phase_frame = running_phase + phase_differences
        new_phases.append(np.reshape(new_phase_frame, (new_phase_frame, 1)))
        running_phase = new_phase_frame
    new_phases = np.hstack(new_phases)
    return new_mags, new_phases

if __name__ == "__main__":
    with pb.io.AudioFile(INPUT_AUDIO, 'r') as in_audio:
        samples = in_audio.read(in_audio.frames)
        sample_rate = in_audio.samplerate

    if samples.ndim > 1:
        samples = np.sum(samples, axis=0)

    stft = signal.ShortTimeFFT(np.hanning(FFT_SIZE), FFT_SIZE // 2, sample_rate)
    specgram = stft.stft(samples)

    mag_specgram = np.abs(specgram)
    phase_specgram = np.angle(specgram)

    new_m, new_p = freeze(mag_specgram, phase_specgram, 8, 50)
    output_specgram = np.multiply(np.cos(new_p), new_m) + 1j * np.multiply(np.sin(new_p), new_m)

    output_audio = stft.istft(output_specgram)

    with pb.io.AudioFile(OUTPUT_AUDIO, 'w', samplerate=sample_rate, num_channels=1) as out_audio:
        out_audio.write(output_audio)
