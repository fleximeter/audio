"""
File: corpus_prep
Date: 10/26/24

Prepares audio files in a corpus so that they are compatible with each other
- Resample to 44.1 kHz
- Normalize audio peak level
- Mix to mono
- Remove DC bias
- Use WAV format
"""

import aus.audiofile as audiofile
import aus.operations as operations
import os
import multiprocessing as mp
import numpy as np
import pathlib
import pedalboard as pb
import re
import scipy.signal

# Directory stuff
IN_DIR = "/Users/jmartin50/recording/samples/granulation_chunks_old"
OUT_DIR = "/Users/jmartin50/recording/samples/granulation_chunks"

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
        out_filename = AUDIO_EXTENSION.sub('', file['out_dir'])
        out_filename = f"{out_filename}.{NEW_EXTENSION}"
        with pb.io.AudioFile(file["in_dir"], 'r').resampled_to(OUT_SAMPLE_RATE) as infile:
            samples = infile.read(infile.frames)
            samples = operations.mixdown(samples)
            if samples.ndim == 1:
                samples = np.reshape(samples, (1, samples.shape[0]))
            samples = operations.adjust_level(samples, -12)
            samples = scipy.signal.sosfilt(LOWCUT_FILTER_COEF, samples)
            with pb.io.AudioFile(os.path.join(OUT_DIR, out_filename), 'w', OUT_SAMPLE_RATE, samples.shape[0], OUT_BIT_DEPTH) as outfile:
                outfile.write(samples)


if __name__ == "__main__":
    print("Converting...")
    
    # Create the output directory
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    root_dir = pathlib.Path(IN_DIR)

    # Find all files
    audio_files = []
    for dir, subdirs, files in os.walk(IN_DIR):
        # Make the output directory if necessary
        out_subdir = os.path.join(OUT_DIR, str(pathlib.Path(dir).relative_to(root_dir)))
        pathlib.Path(out_subdir).mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if AUDIO_EXTENSION.search(file):
                audio_files.append({"in_dir": os.path.join(dir, file), "out_dir": os.path.join(out_subdir, file)})

    # Start the converter processes
    num_processes = mp.cpu_count()
    num_files_per_process = len(audio_files) // num_processes + 1
    processes = [mp.Process(target=file_converter, args=(audio_files[num_files_per_process * i:num_files_per_process * (i + 1)],)) for i in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done")
