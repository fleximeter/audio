"""
File: dev.py

This file is for experimenting.
"""

import audiopython.audiofile as audiofile
import audiopython.operations as operations
import numpy as np
import os
import platform
from fractions import Fraction


OUT_MAC = "/Users/jmartin50/Recording"
OUT_PC = "D:\\Recording"
IN_MAC = "/Volumes/AudioJeff/Recording/ReaperProjects/fixedmedia2/algorithm.wav"
IN_PC = "D:\\Recording\\ReaperProjects\\fixedmedia2\\algorithm.wav"
SYSTEM = platform.system()

if SYSTEM == "Darwin":
    IN = IN_MAC
    OUT = OUT_MAC
else:
    IN = IN_PC
    OUT = OUT_PC

audio = audiofile.read(IN)

beats = [
    Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 3),
    Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 3), Fraction(1, 3),
    Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3),
    Fraction(1, 2), Fraction(1, 2), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3),
    Fraction(1, 2), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3),
]


mapper = [0, 1, 3, 5, 7, 6, 4, 2]

channel_levels = operations.panner(4, 0, 3, 30)
channel_levels = np.repeat(channel_levels, 2, axis=-1)

channel_levels = operations.panner(8, 0.5, 7.5, 30)
channel_levels = operations.pan_mapper(channel_levels, mapper)

NUM_CHANNELS = 8
channel_levels = channel_levels.tolist()
envelope = operations.beat_envelope_multichannel(120, audio.sample_rate, beats, NUM_CHANNELS, channel_levels, "hanning", 8000)
envelope = np.hstack((np.zeros((NUM_CHANNELS, audio.sample_rate // 2)), envelope, np.zeros((NUM_CHANNELS, audio.sample_rate // 2))))
a = audio.samples[0, :envelope.shape[-1]]
a = np.reshape(a, (1, a.shape[-1]))
a = np.repeat(a, NUM_CHANNELS, 0)

a *= envelope
audio.samples = a
audio.num_channels = NUM_CHANNELS
audiofile.write_with_pedalboard(audio, os.path.join(OUT, "test5.wav"))
