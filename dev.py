"""
File: dev.py

This file is for experimenting.
"""

import audiopython.analysis as analysis
import pedalboard as pb
import datetime

FILE = "D:\\Recording\\Samples\\Iowa\\Cello.arco.mono.2444.1\\samples_ff\\sample_Cello.arco.ff.sulA.A3Ab4.wav_7.wav"

with pb.io.AudioFile(FILE, 'r') as infile:
    samples = infile.read(infile.frames)
    samplerate = infile.samplerate

samples = samples[0, 10000:14096]

def analyze(samples, samplerate):
    a = datetime.datetime.now()
    z = analysis.analyzer(samples, samplerate)
    print(datetime.datetime.now() - a)

analyze(samples, samplerate)