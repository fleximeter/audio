"""
File: sample_processor.py
Author: Jeff Martin
Date: 12/22/23

This file loads all samples files within a directory and its subdirectories,
and processes them. It is useful for performing postprocessing after sample extraction,
for naming samples properly and for applying some filtering and tuning.

It is customized for working with University of Iowa EMS samples.
"""

import json
import os
import pedalboard
import platform
import re

# Directory stuff
WINROOT = "D:\\"
MACROOT = "/Volumes/AudioJeff"
PLATFORM = platform.platform()
ROOT = WINROOT

if re.search(r'macos', PLATFORM, re.IGNORECASE):
    ROOT = MACROOT

DIR = os.path.join(ROOT, "Recording", "Compositions", "trombone_piece", "TenorTrombone")


if __name__ == "__main__":
    print("Starting sample processor...")
    destination_directory = os.path.join(DIR, "samples")
    os.makedirs(destination_directory, 511, True)

    with open("sample_processing/config/process.tenortrombone.pp.json", "r") as f:
        data = json.loads(f.read())
        for file in data:
            with pedalboard.io.AudioFile(file["file"], "r") as a:
                audio = a.read(a.frames)
                new_filename = re.sub(r'\.[0-9]+\.wav$', '', os.path.split(file["file"])[-1])
                with pedalboard.io.AudioFile(os.path.join(destination_directory, f"sample.{file['midi']}.{new_filename}.wav"), 'w', 44100, 1, 24) as outfile:
                    outfile.write(audio)

    print("Sample processor done.")
