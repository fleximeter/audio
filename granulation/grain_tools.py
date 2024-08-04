"""
File: grain_tools.py

This file is for granulation tools. You need to build it before using it:
`python setup.py build_ext --inplace
"""

import cython
import numpy as np


def crossfade(audio1: np.ndarray, audio2: np.ndarray, merge_fraction: cython.double):
    """
    Crossfades two audio arrays
    :param audio1: An audio array
    :param audio2: An audio array
    :param merge_fraction: The percentage of overlap for merging. The smallest audio array will be chosen for calculating this percentage.
    :return: The merged audio
    """
    i: cython.int
    j: cython.int
    start_idx: cython.int
    overlap_len = int(min(audio1.shape[-1], audio2.shape[-1]) * merge_fraction)
    x = np.linspace(0, np.pi / 2, overlap_len, False)
    sin_arr = np.sin(x)
    cos_arr = np.cos(x)
    if audio1.ndim == 2:
        new_audio = np.hstack((audio1, np.zeros((audio1.shape[0], audio2.shape[-1] - overlap_len))))
        start_idx = audio1.shape[-1] - overlap_len
        for i in range(audio1.shape[0]):
            for j in range(0, overlap_len):
                new_audio[i, j + start_idx] = new_audio[i, j + start_idx] * cos_arr[j] + audio2[i, j] * sin_arr[j]
            for j in range(overlap_len, audio2.shape[-1]):
                new_audio[i, j + start_idx] = audio2[i, j]
    elif audio1.ndim == 1:
        new_audio = np.hstack((audio1, np.zeros((audio2.shape[-1] - overlap_len))))
        start_idx = audio1.shape[-1] - overlap_len
        for j in range(0, overlap_len):
            new_audio[j + start_idx] = new_audio[j + start_idx] * cos_arr[j] + audio2[j] * sin_arr[j]
        for j in range(overlap_len, audio2.shape[-1]):
            new_audio[j + start_idx] = audio2[j]
    return new_audio
   

def merge_grain(audio: np.ndarray, grain: np.ndarray, start_idx: cython.int, end_idx: cython.int, channel: cython.int):
    """
    Merges a grain array into an audio array
    :param audio: The audio array
    :param grain: The grain
    :param start_idx: The start index for merging
    :param end_idx: The end index for merging
    :param channel: The channel in which to merge
    """
    i: cython.int
    j: cython.int
    i = 0
    j = 0
    if channel == 0 and audio.ndim == 1:
        for i in range(start_idx, end_idx):
            audio[i] += grain[j]
            j += 1
    else:
        for i in range(start_idx, end_idx):
            audio[channel, i] += grain[j]
            j += 1
