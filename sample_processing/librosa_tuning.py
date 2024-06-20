"""
File: librosa_tuning.py
Author: Jeff Martin
Date: 6/20/24

This file contains functionality for tuning with librosa.
"""

import librosa
import numpy as np


def librosa_pitch_estimation(audio, sample_rate=44100, min_freq=55, max_freq=880, quantile=0.5):
    """
    Estimates the pitch of the signal, based on the LibRosa pyin function
    :param audio: A NumPy array of audio samples
    :param sample_rate: The sample rate of the audio
    :param min_freq: The minimum frequency allowed for the pyin function
    :param max_freq: The maximum frequency allowed for the pyin function
    :param quantile: The quantile to select the frequency from, since the frequencies 
    are calculated as an array of frequencies. Normally the median (0.5) is a good choice.
    :return: The pitch
    """
    estimates = librosa.pyin(audio, fmin=min_freq, fmax=max_freq, sr=sample_rate)
    nans = set()
    for i in range(estimates[0].shape[-1]):
        if np.isnan(estimates[0][i]) or np.isinf(estimates[0][i]) or np.isneginf(estimates[0][i]):
            nans.add(i)
    # We arbitrarily decide that if half of the detected pitches are NaN, we will
    # be returning NaN
    if estimates[0].shape[-1] // 2 > len(nans):
        for i in nans:
            estimates[0][i] = 0
    return np.quantile(estimates[0], quantile)


def midi_estimation_from_pitch(frequency):
    """
    Estimates MIDI note number from provided frequency
    :param frequency: The frequency
    :return: The midi note number (or NaN)
    """
    midi_est = 12 * np.log2(frequency / 440) + 69
    if np.isnan(midi_est) or np.isneginf(midi_est) or np.isinf(midi_est):
        midi_est = 0.0
    return midi_est


def midi_tuner(audio: np.array, midi_estimation, midi_division=1, sample_rate=44100, target_midi=None) -> np.array:
    """
    Retunes audio from a provided midi estimation to the nearest accurate MIDI note
    :param audio: The audio to tune
    :param midi_estimation: The MIDI estimation
    :param midi_division: The MIDI division to tune to (1 for nearest semitone, 0.5 for nearest quarter tone)
    :param sample_rate: The sample rate of the audio
    :param target_midi: If specified, overrides the rounding functionality and uses this as the target MIDI note
    :return: The tuned audio
    """
    if not target_midi:
        target_midi = round(float(midi_estimation / midi_division)) * midi_division
    ratio = 2 ** ((target_midi - midi_estimation) / 12)
    new_sr = sample_rate * ratio
    # print(midi_estimation, new_midi, ratio, new_sr)
    return librosa.resample(audio, orig_sr=new_sr, target_sr=sample_rate, res_type="soxr_vhq")

