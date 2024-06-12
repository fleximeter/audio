"""
File: pedalboard_file_converter
Date: 12/19/23

A pedalboard-based file converter. This is for doing things like changing audio format
from AIFF to WAV, along with doing cool things like HPF for eliminating DC bias.
"""

import audiopython.audiofile as audiofile
import audiopython.operations as operations
import os
import multiprocessing as mp
import pathlib
import pedalboard as pb
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

IN_DIR = os.path.join(ROOT, "Recording", "Samples", "Iowa", "Viola.arco.mono.2444.1")
OUT_DIR = os.path.join(ROOT, "Recording", "Samples", "Iowa", "Viola.arco.mono.2444.1", "process")

# Basic audio stuff
LOWCUT_FREQ = 20
OUT_SAMPLE_RATE = 44100
OUT_BIT_DEPTH = 24
NEW_EXTENSION = "wav"

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
audio_files = audiofile.find_files(IN_DIR)
subber = re.compile(r'(\.aif+$)|(\.wav$)')

# The filter we use to remove DC bias and any annoying low frequency stuff
filt = scipy.signal.butter(8, LOWCUT_FREQ, 'high', output='sos', fs=OUT_SAMPLE_RATE)


def file_converter_resample(files):
    for file in files:
        filename = os.path.split(file)[1]
        filename = subber.sub('', filename)
        filename = f"{filename}.{NEW_EXTENSION}"
        with pb.io.AudioFile(file, 'r').resampled_to(OUT_SAMPLE_RATE) as infile:
            with pb.io.AudioFile(os.path.join(OUT_DIR, filename), 'w', OUT_SAMPLE_RATE, infile.num_channels, OUT_BIT_DEPTH) as outfile:
                while infile.tell() < infile.frames:
                    outfile.write(infile.read(1024))


def file_converter_resample_filter(files):
    for file in files:
        filename = os.path.split(file)[1]
        filename = subber.sub('', filename)
        filename = f"{filename}.{NEW_EXTENSION}"
        with pb.io.AudioFile(file, 'r').resampled_to(OUT_SAMPLE_RATE) as infile:
            audio = infile.read(infile.frames)
            audio = scipy.signal.sosfilt(filt, audio)
            with pb.io.AudioFile(os.path.join(OUT_DIR, filename), 'w', OUT_SAMPLE_RATE, infile.num_channels, OUT_BIT_DEPTH) as outfile:
                outfile.write(audio)


def file_converter_filter(files):
    for file in files:
        filename = os.path.split(file)[1]
        filename = subber.sub('', filename)
        filename = f"{filename}.{NEW_EXTENSION}"
        with pb.io.AudioFile(file, 'r') as infile:
            audio = infile.read(infile.frames)
            audio = operations.mix_if_not_mono(audio)
            audio = scipy.signal.sosfilt(filt, audio)
            with pb.io.AudioFile(os.path.join(OUT_DIR, filename), 'w', OUT_SAMPLE_RATE, 1, OUT_BIT_DEPTH) as outfile:
                outfile.write(audio)


if __name__ == "__main__":
    print("Converting...")
    num_processes = mp.cpu_count()
    num_files_per_process = len(audio_files) // num_processes + 1
    processes = [mp.Process(target=file_converter_filter, args=(audio_files[num_files_per_process * i:num_files_per_process * (i + 1)],)) for i in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done")
