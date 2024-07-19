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
    
    effect_chain = [
        ButterworthFilterEffect(50, "highpass", 4)
    ]
    effect_cycle = [
        IdentityEffect(), 
        ChorusEffect(2, 0.5, 20, 0.4, 0.5),
        IdentityEffect(), 
        ButterworthFilterEffect(440, "lowpass", 2),
        IdentityEffect(), 
        ButterworthFilterEffect(440, "lowpass", 2),
        IdentityEffect(), 
        ChorusEffect(2, 0.5, 20, 0.4, 0.5),
    ]

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
            AND (length = 8192)
            AND (spectral_roll_off_75 < 500)
            AND (energy > 0.2);
    """

    # Retrieve grain metadata and grains
    db, cursor = grain_sql.connect_to_db(DB_FILE)
    grains1 = grain_sql.realize_grains(cursor, SELECT, SOURCE_DIR)
    print(f"{len(grains1)} grains found.")
    db.close()

    # Assemble grains
    # samples = assemble_single(grains1, ["spectral_roll_off_50", "spectral_slope_0_1_khz"], 3000, -18.0, np.hanning, effect_chain, effect_cycle)
    # samples = grain_assembler.assemble_stochastic(grains1, 5, 3000, rng, -18.0, np.hanning, effect_chain, effect_cycle)

    # Generate candidate audio
    NUM_CANDIDATES = 5
    for i in range(NUM_CANDIDATES):
        LIST_LENGTH = 10
        grain_list = []
        for j in range(LIST_LENGTH):
            grain_list.append(grains1[rng.randrange(0, len(grains1))])

        samples = grain_assembler.repeat(grain_list, 50, -7000, -18.0, np.hanning, effect_chain, effect_cycle)
        
        # Apply final effects to the assembled audio
        lpf = signal.butter(2, 500, btype="lowpass", output="sos", fs=44100)
        hpf = signal.butter(8, 100, btype="highpass", output="sos", fs=44100)
        samples = signal.sosfilt(lpf, samples)
        samples = signal.sosfilt(hpf, samples)
        samples = operations.fade_in(samples, "hanning", 22050)
        samples = operations.fade_out(samples, "hanning", 22050)
        samples = operations.adjust_level(samples, -12)

        # Write the audio
        audio = audiofile.AudioFile(sample_rate=44100, bits_per_sample=24, num_channels=1)
        audio.samples = samples
        audiofile.write_with_pedalboard(audio, f"D:\\Recording\\temp6_{i+1}.wav")
