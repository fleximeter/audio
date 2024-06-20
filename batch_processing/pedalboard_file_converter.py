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

IN_DIR = os.path.join(ROOT, "Recording", "Compositions", "trombone_piece", "TenorTrombone")
OUT_DIR = os.path.join(ROOT, "Recording", "Compositions", "trombone_piece", "TenorTrombone", "process")

# Basic audio stuff
LOWCUT_FREQ = 10
OUT_SAMPLE_RATE = 44100
OUT_BIT_DEPTH = 24
NEW_EXTENSION = "wav"

# Used to make sure we only work with audio files; also for removing the extension as needed
AUDIO_EXTENSION = re.compile(r'(\.aif+$)|(\.wav$)', re.IGNORECASE)

# The filter we use to remove DC bias and any annoying low frequency stuff
LOWCUT_FILTER_COEF = scipy.signal.butter(8, LOWCUT_FREQ, 'high', output='sos', fs=OUT_SAMPLE_RATE)


def file_converter_resample(files):
    """
    Converts and resamples all files to the given sample rate
    :param files: A list of files
    """
    for file in files:
        filename = os.path.split(file)[1]
        filename = AUDIO_EXTENSION.sub('', filename)
        filename = f"{filename}.{NEW_EXTENSION}"
        with pb.io.AudioFile(file, 'r').resampled_to(OUT_SAMPLE_RATE) as infile:
            with pb.io.AudioFile(os.path.join(OUT_DIR, filename), 'w', OUT_SAMPLE_RATE, infile.num_channels, OUT_BIT_DEPTH) as outfile:
                while infile.tell() < infile.frames:
                    outfile.write(infile.read(1024))


def file_converter_resample_filter(files):
    """
    Converts all files to the given sample rate and adds a highpass filter to remove DC offset
    :param files: A list of files
    """
    for file in files:
        filename = os.path.split(file)[1]
        filename = AUDIO_EXTENSION.sub('', filename)
        filename = f"{filename}.{NEW_EXTENSION}"
        with pb.io.AudioFile(file, 'r').resampled_to(OUT_SAMPLE_RATE) as infile:
            audio = infile.read(infile.frames)
            audio = scipy.signal.sosfilt(LOWCUT_FILTER_COEF, audio)
            with pb.io.AudioFile(os.path.join(OUT_DIR, filename), 'w', OUT_SAMPLE_RATE, infile.num_channels, OUT_BIT_DEPTH) as outfile:
                outfile.write(audio)


def file_converter_filter(files):
    """
    Converts all files and adds a highpass filter to remove DC offset
    :param files: A list of files
    """
    for file in files:
        filename = os.path.split(file)[1]
        filename = AUDIO_EXTENSION.sub('', filename)
        filename = f"{filename}.{NEW_EXTENSION}"
        with pb.io.AudioFile(file, 'r') as infile:
            audio = infile.read(infile.frames)
            audio = operations.mixdown(audio)
            audio = scipy.signal.sosfilt(LOWCUT_FILTER_COEF, audio)
            with pb.io.AudioFile(os.path.join(OUT_DIR, filename), 'w', OUT_SAMPLE_RATE, 1, OUT_BIT_DEPTH) as outfile:
                outfile.write(audio)


if __name__ == "__main__":
    print("Converting...")
    
    # Create the output directory
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Find all files
    audio_files = []
    for dir, subdirs, files in os.walk(IN_DIR):
        for file in files:
            if AUDIO_EXTENSION.search(file):
                audio_files.append(os.path.join(dir, file))

    # Start the converter processes
    num_processes = mp.cpu_count()
    num_files_per_process = len(audio_files) // num_processes + 1
    processes = [mp.Process(target=file_converter_filter, args=(audio_files[num_files_per_process * i:num_files_per_process * (i + 1)],)) for i in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done")
