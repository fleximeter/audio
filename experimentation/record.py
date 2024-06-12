"""
File: record.py
Author: Jeff Martin
Date: 11/29/23

This file records audio to a NumPy array.
"""

import sounddevice as sd

# To get a list of sound devices:
# sd.query_devices()

sd.default.device = 14
sd.default.channels = 1
sd.default.samplerate = 44100

# this can be saved to file with pedalboard
buffer = sd.rec(int(5 * sd.default.samplerate), channels=1, dtype="float64")
print(buffer[40000:40100])

sd.play(buffer)
