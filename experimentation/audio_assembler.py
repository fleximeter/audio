"""
File: audio_assembler.py

This file is for assembling multichannel audio files from individual channel files.
"""

import aus.audiofile as audiofile
import numpy as np
import os

if __name__ == "__main__":
    # Make sure that all the files are in this directory, and that no other files are in this
    # directory or its subdirectories!
    # Also, make sure that the files are named in order that you want their channels.
    # You can number them, or use any scheme, as long as they will be ordered properly after sorting.
    # And make sure all audio files have the same sample rate.
    DIR = "D:\\Recording\\Temp"

    # Set the output file name here
    OUT = os.path.join(DIR, "out.wav")
    
    files = audiofile.find_files(DIR)
    files.sort()
    audiofiles = [audiofile.read(file) for file in files]
    samples = [f.samples for f in audiofile]
    max_len = 0
    for i in range(len(samples)):
        if samples[i].shape[-1] > max_len:
            max_len = samples[i].shape[-1]
    for i in range(len(samples)):
        if samples[i].shape[-1] < max_len:
            shape = list(samples[i].shape)
            shape[-1] = max_len - shape[-1]
            samples[i] = np.hstack((samples[i], np.zeros(shape, dtype=samples[i].dtype)))
    samples = np.vstack(samples)
    newfile = audiofile.AudioFile(bits_per_sample=24, num_channels=samples.shape[0], sample_rate=audiofiles[0].sample_rate)
    newfile.samples = samples
    audiofile.write_with_pedalboard(newfile, OUT)
