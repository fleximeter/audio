"""
File: audio_splitter.py

This file is for splitting multichannel audio files into individual channel files.
"""

import aus.audiofile as audiofile
import numpy as np
import os
import re

if __name__ == "__main__":
    # Make sure that all the files are in this directory, and that no other files are in this
    # directory or its subdirectories!
    # Also, make sure that the files are named in order that you want their channels.
    # You can number them, or use any scheme, as long as they will be ordered properly after sorting.
    # And make sure all audio files have the same sample rate.
    DIR = "/Users/jmartin50/recording/out"

    # Set the output file name here
    OUT = "/Users/jmartin50/recording/out/split"

    AUDIO_EXTENSION = re.compile(r'(\.aif+$)|(\.wav$)', re.IGNORECASE)

    files = audiofile.find_files(DIR)
    files.sort()
    print("Splitting audio channels into separate files...")
    for file in files:
        a = audiofile.read(file)
        filename = os.path.split(file)[-1]
        print(filename)
        filename = AUDIO_EXTENSION.sub('', filename)
        for i in range(a.num_channels):
            newfile = audiofile.AudioFile(sample_rate=a.sample_rate, num_channels=1, frames=a.frames, bits_per_sample=a.bits_per_sample)
            newfile.samples = np.reshape(a.samples[i, :], (1, a.frames))
            audiofile.write_with_pedalboard(newfile, os.path.join(OUT, f"{filename}_{i+1}.wav"))
    print("Done.")
