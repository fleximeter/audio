"""
File: dev.py

This file is for experimenting.
"""

import audiopython.audiofile as audiofile
import audiopython.operations as operations
import audiopython.synthesis as synthesis
import pedalboard as pb
import datetime
import numpy as np
from fractions import Fraction

FILE = "D:\\Recording\\ReaperProjects\\fixedmedia2\\algorithm.wav"

audio = audiofile.read(FILE)

beats = [
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 2),
    Fraction(1, 3),
]

channel_levels = [(0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), 
                  (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), 
                  (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), 
                  (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 0), 
                  (0, 1), (0, 1), (1, 0), (1, 0), (0, 1), (0, 1)]

NUM_CHANNELS = 2

envelope = operations.beat_envelope_multichannel(120, audio.sample_rate, beats, NUM_CHANNELS, channel_levels, "hanning", 8000)
envelope = np.hstack((np.zeros((NUM_CHANNELS, audio.sample_rate // 2)), envelope, np.zeros((NUM_CHANNELS, audio.sample_rate // 2))))
a = audio.samples[0, :envelope.shape[-1]]
a = np.reshape(a, (1, a.shape[-1]))
a = np.repeat(a, NUM_CHANNELS, 0)

a *= envelope
audio.samples = a
audio.num_channels = NUM_CHANNELS
audiofile.write_with_pedalboard(audio, "D:\\Recording\\test1.wav")
