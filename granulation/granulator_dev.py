"""
File: granulator_dev.py

This file is for experimenting with granulation.
"""

import grain_sql
import aus.audiofile as audiofile
import aus.operations as operations
import random
import scipy.signal as signal
from effects import *
import grain_assembler
import grain_sql


if __name__ == "__main__":
    rng = random.Random()
    rng.seed()
    
    # The directory containing the files that were analyzed. We can search in here for the files,
    # even if the path doesn't match exactly. This is needed because we may have performed
    # the analysis on a different computer.
    SOURCE_DIR = "D:\\Recording\\Samples\\freesound\\creative_commons_0\\granulation"
    
    # The database
    DB_FILE = "D:\\Source\\grain_processor\\data\\grains.sqlite3"
    SELECT = """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness < 0.1) 
            AND (length = 4096)
            AND (spectral_roll_off_75 < 500)
            AND (energy > 0.2);
    """
    db, cursor = grain_sql.connect_to_db(DB_FILE)
    grains1 = grain_sql.realize_grains(cursor, SELECT, SOURCE_DIR)
    if len(grains1) > 1000:
        grains1 = grains1[:1000]
    print(f"{len(grains1)} grains found.")
    audio = audiofile.AudioFile(sample_rate=44100, bits_per_sample=24, num_channels=1)
    # samples = assemble_single(grains1, ["spectral_roll_off_50", "spectral_slope_0_1_khz"], 3000)
    samples = grain_assembler.assemble_stochastic(grains1, 5, 3000, rng)
    
    lpf = signal.butter(2, 500, btype="lowpass", output="sos", fs=44100)
    hpf = signal.butter(8, 100, btype="highpass", output="sos", fs=44100)
    samples = signal.sosfilt(lpf, samples)
    samples = signal.sosfilt(hpf, samples)
    samples = operations.fade_in(samples, "hanning", 22050)
    samples = operations.fade_out(samples, "hanning", 22050)
    samples = operations.adjust_level(samples, -12)
    audio.samples = samples

    audiofile.write_with_pedalboard(audio, "D:\\Recording\\temp4.wav")
    db.close()
