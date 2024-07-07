"""
File: dev.py

This file is for experimenting.
"""

import audiopython.audiofile as audiofile
import audiopython.operations as operations
import audiopython.synthesis as synthesis
import pedalboard as pb
import datetime

sample_rate = 44100
length = sample_rate * 5
sig1 = synthesis.saw(220, 20, length, sample_rate)
sig2 = synthesis.saw(220 * 5/4, 20, length, sample_rate)
sig3 = synthesis.saw(220 * 3/2, 20, length, sample_rate)
sig = sig1 + sig2 + sig3
sig = operations.adjust_level(sig, -12.0)
sig = operations.fade_in(sig, "hanning", sample_rate // 4)
sig = operations.fade_out(sig, "hanning", sample_rate // 4)
audio = audiofile.AudioFile(audio_format=1, bits_per_sample=24, num_channels=1, num_frames=length, sample_rate=sample_rate)
audio.samples = sig
audiofile.write_with_pedalboard(audio, "D:\\Recording\\test.wav")
