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


def force_equal_energy(audio: np.ndarray, dbfs: float = -6.0, window_size: int = 8192, max_scalar: float = 1e6):
    """
    Forces equal energy on a mono signal over time. For example, if a signal initially has high energy, 
    and gets less energetic, this will adjust the energy level so that it does not decrease.
    Better results come with using a larger window size, so the energy changes more gradually.
    :param audio: The array of audio samples
    :param dbfs: The target level of the entire signal, in dbfs
    :param window_size: The window size to consider when detecting RMS energy
    :param max_scalar: The maximum scalar to use in level adjustment. This is necessary
    because some audio may have an extremely low maximum level, and the scalar required
    to bring it up to the right level could result in computation problems. The computed
    level scalar used in this function will be -max_scalar <= level <= max_scalar.
    :return: An adjusted version of the signal
    """
    i: cython.int
    j: cython.int
    k: cython.int
    idx: cython.int
    frame_idx: cython.int
    if audio.ndim == 1:
        raise Exception("The audio array must have two dimensions.")
    num_channels = audio.shape[0]
    output_audio_arr = np.empty(audio.shape)  # the new array we'll be returning
    target_level = 10 ** (dbfs / 20)  # the target level, in float rather than dbfs
    num_frames = int(np.ceil(audio.shape[-1] / window_size))  # the number of frames that we'll be analyzing
    energy_levels = np.empty((num_channels, num_frames + 2))  # the energy level for each frame
    
    # find the energy levels
    for i in range(num_channels):
        idx = 1
        for j in range(0, audio.shape[-1], window_size):
            end_idx = min(j+window_size, audio.shape[-1])
            energy_levels[i, idx] = np.sqrt(np.average(np.square(audio[i, j:end_idx])))
            idx += 1
        energy_levels[i, 0] = energy_levels[i, 1]
        energy_levels[i, -1] = energy_levels[i, -2]

    # do the first half frame
    for i in range(num_channels):
        coef = 1 / energy_levels[i, 0]
        coef = max(1/max_scalar, coef)
        coef = min(max_scalar, coef)
        for j in range(0, window_size // 2):
            output_audio_arr[i, j] = audio[i, j] * coef
    
    # do adjacent half frames from 1 and 2, 2 and 3, etc.
    for i in range(num_channels):
        frame_idx = 1
        for frame_start_idx in range(window_size // 2, audio.shape[-1], window_size):
            end_idx = min(frame_start_idx + window_size, audio.shape[-1])
            frame_size = end_idx - frame_start_idx
            slope = (energy_levels[i, frame_idx + 1] - energy_levels[i, frame_idx]) / frame_size
            y_int = energy_levels[i, frame_idx]
            for sample_idx in range(frame_start_idx, end_idx):
                f_x = slope * (sample_idx - frame_start_idx) + y_int
                scalar = 1/f_x
                scalar = max(1/max_scalar, scalar)
                scalar = min(max_scalar, scalar)
                output_audio_arr[i, sample_idx] = audio[i, sample_idx] * scalar
            frame_idx += 1

    audio_max = np.max(np.abs(output_audio_arr))
    scalar = target_level / audio_max
    scalar = max(1/max_scalar, scalar)
    scalar = min(max_scalar, scalar)
    return output_audio_arr * scalar
    

x = synthesis.sine(440, 0, 44100 * 10, 44100) * 0.5
for i in range(44100, 88200):
    x[i] *= 0.5
# x = operations.fade_out(x, "hanning", 44100 * 10)
x = x.reshape((1, x.shape[0]))
x = force_equal_energy(x, -6, 8192)
# x = operations.fade_in(x, "hanning", 11025)
# x = operations.fade_out(x, "hanning", 11025)
a = audiofile.AudioFile(bits_per_sample=16, sample_rate=44100, num_channels=1, num_frames=44100*10)
a.samples = x
audiofile.write_with_pedalboard(a, "D:\\Recording\\out\\test.wav")
