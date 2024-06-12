"""
File: sample_processor.py
Author: Jeff Martin
Date: 12/22/23

This file loads all samples files within a directory and its subdirectories,
and processes them. It is useful for performing postprocessing after sample extraction,
for naming samples properly and for applying some filtering and tuning.

It is customized for working with University of Iowa EMS samples.
"""

import audiopython.analysis as analysis
import audiopython.operations as operations
import json
import numpy as np
import os
import pedalboard
import platform
import re
import scipy.signal

# Directory stuff
WINROOT = "D:\\"
MACROOT = "/Volumes/AudioJeff"
PLATFORM = platform.platform()
ROOT = WINROOT

if re.search(r'macos', PLATFORM, re.IGNORECASE):
    ROOT = MACROOT

DIR = os.path.join(ROOT, "Recording", "Samples", "Iowa", "Xylophone.rosewood")


if __name__ == "__main__":
    print("Starting sample processor...")
    destination_directory = os.path.join(DIR, "samples")
    os.makedirs(destination_directory, 511, True)

    with open("config/process.xylophone.rosewood.ff.json", "r") as f:
        data = json.loads(f.read())
        for file in data:
            with pedalboard.io.AudioFile(file["file"], "r") as a:
                audio = a.read(a.frames)
                audio = scipy.signal.sosfilt(
                    scipy.signal.butter(12, 440 * 2 ** ((file["midi"] - 5 - 69) / 12), 'high', output='sos', fs=44100), 
                    audio
                    )
                midi_est = analysis.midi_estimation_from_pitch(
                    analysis.librosa_pitch_estimation(
                        operations.mix_if_not_mono(audio), 
                        44100, 
                        440 * 2 ** ((file["midi"] - 4 - 69) / 12), 
                        440 * 2 ** ((file["midi"] + 4 - 69) / 12), 
                        0.5
                    ))
                if not np.isnan(midi_est) and not np.isinf(midi_est) and not np.isneginf(midi_est):
                   audio = operations.midi_tuner(audio, midi_est, 1, 44100, file["midi"])
                new_filename = re.sub(r'\.[0-9]+\.wav$', '', os.path.split(file["file"])[-1])
                with pedalboard.io.AudioFile(os.path.join(destination_directory, f"sample.{file['midi']}.{new_filename}.wav"), 'w', 44100, 1, 24) as outfile:
                    outfile.write(audio)

    print("Sample processor done.")
