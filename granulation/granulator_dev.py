"""
File: granulator_dev.py

This file is for experimenting with granulation.
"""

import audiopython.audiofile as audiofile
import audiopython.granulator as granulator
import audiopython.operations as operations
import audiopython.analysis as analysis
import audiopython.sampler as sampler
import os
import pedalboard as pb
import platform
import re
import numpy as np
import random

random.seed()

np.seterr(divide="ignore")

# Directory stuff
WINROOT = "D:\\"
MACROOT = "/Volumes/AudioJeff"
PLATFORM = platform.platform()
ROOT = WINROOT

if re.search(r'macos', PLATFORM, re.IGNORECASE):
    ROOT = MACROOT

DIR = os.path.join(ROOT, "Recording", "Samples", "Iowa", "Viola.arco.mono.2444.1", "samples")
DIR_OUT = os.path.join(ROOT, "Recording")

# Basic audio stuff
NUM_GRAINS = 20000
GRAIN_SIZE_MIN = 1000
GRAIN_SIZE_MAX = 10000

samples = sampler.load_samples_iowa(DIR)
grains = []

for i in range(NUM_GRAINS):
    grain_size = random.randint(GRAIN_SIZE_MIN, GRAIN_SIZE_MAX)
    j = random.randint(0, len(samples)-1)
    start_frame = random.randint(0, samples[j].samples.shape[-1] - grain_size)
    grain = granulator.extract_grain(samples[j].samples, start_frame, grain_size, max_window_size=GRAIN_SIZE_MIN)
    grain1 = sampler.Sample(grain, samples[j].sample_rate, samples[j].path)
    grain1.midi = samples[j].midi
    grain1.instrument_name = samples[j].instrument_type
    grain1.dynamic_name = samples[j].dynamic_name
    grain1.dynamic_id = samples[j].dynamic_id
    grain1.pitched = samples[j].pitched
    grains.append(grain1)

granulator.scale_grain_peaks(grains)
max_dbfs = granulator.find_max_grain_dbfs(grains)
grains2 = []
for grain in grains:
    if operations.dbfs_audio(grain.samples) / max_dbfs < 2:
        grain.analysis = analysis.analyzer(grain.samples, grain.sample_rate)
        grains2.append(grain)

grains2 = sorted(grains2, key=lambda x: x.analysis["spectral_roll_off_0.5"])
grains2 = sorted(grains2, key=lambda x: x.midi)

final_audio = granulator.merge_grains(grains2, GRAIN_SIZE_MIN // 2)
final_audio = pb.LowpassFilter(4000)(final_audio, 44100)
final_audio = pb.Compressor(-12, 4)(final_audio, 44100)

with pb.io.AudioFile(f"{DIR_OUT}\\grains.wav", 'w', 44100, 1, 24) as out:
    out.write(final_audio)
