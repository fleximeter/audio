"""
File: grain_tools.py

This file is for granulation tools.
"""

import cython
import numpy as np


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
