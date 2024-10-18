"""
File: audio_normalizer
Date: 10/17/24

Normalizes audio files to a specific peak dB
"""

import aus.audiofile as audiofile
import aus.operations as operations
import os
import multiprocessing as mp
import pathlib
import pedalboard as pb
import platform
import re
import scipy.signal

# Directory stuff
IN_DIR = "D:\\Recording\\ReaperProjects\\viola_piece\\Media\\samples"
OUT_DIR = "D:\\Recording\\ReaperProjects\\viola_piece\\Media\\samples\\out"

# Basic audio stuff
LOWCUT_FREQ = 10
OUT_SAMPLE_RATE = 44100
OUT_BIT_DEPTH = 24
NEW_EXTENSION = "wav"

# Used to make sure we only work with audio files; also for removing the extension as needed
AUDIO_EXTENSION = re.compile(r'(\.aif+$)|(\.wav$)', re.IGNORECASE)

# The filter we use to remove DC bias and any annoying low frequency stuff
LOWCUT_FILTER_COEF = scipy.signal.butter(8, LOWCUT_FREQ, 'high', output='sos', fs=OUT_SAMPLE_RATE)


def file_converter(files):
    """
    Converts all files
    :param files: A list of files
    """
    for file in files:
        filename = os.path.split(file)[1]
        filename = AUDIO_EXTENSION.sub('', filename)
        filename = f"{filename}.{NEW_EXTENSION}"
        with pb.io.AudioFile(file, 'r') as infile:
            samples = infile.read(infile.frames)
            samples = operations.adjust_level(samples, -12)
            with pb.io.AudioFile(os.path.join(OUT_DIR, filename), 'w', OUT_SAMPLE_RATE, infile.num_channels, OUT_BIT_DEPTH) as outfile:
                outfile.write(samples)


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
    processes = [mp.Process(target=file_converter, args=(audio_files[num_files_per_process * i:num_files_per_process * (i + 1)],)) for i in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done")
