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
import math


def panner(num_channels, start_pos, end_pos, num_iterations):
    """
    Equal power panner, moving from start_pos to end_pos over num_iterations.
    :param num_channels: The number of channels
    :param start_pos: The start panning position
    :param end_pos: The end panning position
    :param num_iterations: The number of steps to take to move from start_pos to end_pos
    :return: An array of pan coefficients
    """
    pos_arr = np.linspace(start_pos, end_pos, num_iterations)
    for i in range(pos_arr.shape[-1]):
        pos_arr[i] %= num_channels
    pan_coefficients = []
    for i in range(pos_arr.shape[-1]):
        frac, pos = math.modf(float(pos_arr[i]))
        pos = int(pos)
        coefficients = [0 for i in range(num_channels)]

        # Equal power panning
        coefficients[pos] = np.cos(np.pi * frac / 2)
        if pos+1 < num_channels:
            coefficients[pos+1] = np.sin(np.pi * frac / 2)
        
        pan_coefficients.append(coefficients)
    return pan_coefficients


def pan_mapper(pan_coefficients, mapper):
    """
    Maps pan positions to the actual speakers. 
    This is useful if you want to use a different numbering system for your 
    pan positions than the numbering system used for the actual output channels.
    For example, you might want to pan in a circle for a quad-channel setup,
    but the hardware is set up for stereo pairs.
    :param pan_coefficients: A list of pan coefficient lists
    :param mapper: The mapper for reordering the pan coefficients
    :return: A new, mapped pan coefficient list
    """
    newlist = []
    for i in range(len(pan_coefficients)):
        coefficient_arr = []
        for pos in mapper:
            coefficient_arr.append(pan_coefficients[i][pos])
        newlist.append(coefficient_arr)
    return newlist


FILE = "D:\\Recording\\ReaperProjects\\fixedmedia2\\algorithm.wav"

audio = audiofile.read(FILE)

beats = [
    Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 3),
    Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 3), Fraction(1, 3),
    Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3),
    Fraction(1, 2), Fraction(1, 2), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3),
    Fraction(1, 2), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3),
]


mapper = [0, 1, 3, 5, 7, 6, 4, 2]

channel_levels = panner(4, 0, 3, 30)
for i in range(len(channel_levels)):
    for j in range(1, 8, 2):
        channel_levels[i].insert(j, channel_levels[i][j-1])

channel_levels = panner(8, 0.5, 7.5, 30)
channel_levels = pan_mapper(channel_levels, mapper)

NUM_CHANNELS = 8

envelope = operations.beat_envelope_multichannel(120, audio.sample_rate, beats, NUM_CHANNELS, channel_levels, "hanning", 8000)
envelope = np.hstack((np.zeros((NUM_CHANNELS, audio.sample_rate // 2)), envelope, np.zeros((NUM_CHANNELS, audio.sample_rate // 2))))
a = audio.samples[0, :envelope.shape[-1]]
a = np.reshape(a, (1, a.shape[-1]))
a = np.repeat(a, NUM_CHANNELS, 0)

a *= envelope
audio.samples = a
audio.num_channels = NUM_CHANNELS
audiofile.write_with_pedalboard(audio, "D:\\Recording\\test3.wav")
