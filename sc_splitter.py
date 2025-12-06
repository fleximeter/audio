"""
A file for splitting SuperCollider multichannel files into mono audio files.
Performs normalization to eliminate clipping due to floating point excess.
"""

import pedalboard as pb
import numpy as np
import os

PATH = "D:\\Recording\\Projects\\just_bright\\out"

with pb.io.AudioFile(os.path.join(PATH, "out_concert.wav"), 'r') as sc_file:
    print("File metadata:")
    print(f"Num channels: {sc_file.num_channels}")
    print(f"Sample rate: {sc_file.samplerate}")
    samplerate = sc_file.samplerate
    print("Reading audio samples...")
    samples = sc_file.read(sc_file.frames)
    print("Audio samples read.")
dur = samples.shape[-1] / samplerate
minutes = int(dur // 60)
seconds = round(dur - (minutes * 60), 3)
print(f"File duration: {minutes:02}:{seconds:02}")
max_level = np.max(np.abs(samples))
print(f"Max level: {round(20 * np.log10(max_level), 3)} dBFS")
if max_level > 1.0:
    print("Level will be adjusted to prevent clipping.")
    samples = samples / (max_level + 1e-3)
for i in range(samples.shape[0]):
    print(f"Writing file {i+1} of {samples.shape[0]}...")
    with pb.io.AudioFile(os.path.join(PATH, f"out{i+1}.wav"), 'w', samplerate, 1, 24) as out_file:
        out_file.write(samples[i, :])