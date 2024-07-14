"""
File: basic_synthesis.py

This file demonstrates basic sound wave shapes
"""

import audiopython.audiofile as audiofile
import audiopython.operations as operations
import audiopython.synthesis as synthesis
import os
import platform
import numpy as np

OUT_MAC = "/Users/jmartin50/Recording"
OUT_PC = "D:\\Recording"
SYSTEM = platform.system()

if SYSTEM == "Darwin":
    OUT = OUT_MAC
else:
    OUT = OUT_PC

sample_rate = 44100
length = sample_rate * 5

sig1 = synthesis.sine(220, np.pi / 3, length, sample_rate)
sig2 = synthesis.sine(220 * 5/4, 9 * np.pi / 5, length, sample_rate)
sig3 = synthesis.sine(220 * 3/2, 3 * np.pi / 2, length, sample_rate)
sig = sig1 + sig2 + sig3
sig = operations.adjust_level(sig, -12.0)
sig = operations.fade_in(sig, "hanning", sample_rate // 4)
sig = operations.fade_out(sig, "hanning", sample_rate // 4)
audio = audiofile.AudioFile(audio_format=1, bits_per_sample=24, num_channels=1, num_frames=length, sample_rate=sample_rate)
audio.samples = sig
audiofile.write_with_pedalboard(audio, os.path.join(OUT, "test.wav"))
