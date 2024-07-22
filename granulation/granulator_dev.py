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
        IdentityEffect(), 
        ChorusEffect(2, 0.5, 20, 0.4, 0.5),
        ButterworthFilterEffect(440, "lowpass", 2),
        IdentityEffect(), 
        IdentityEffect(), 
        ButterworthFilterEffect(440, "lowpass", 2),
        ChorusEffect(2, 0.5, 20, 0.4, 0.5),
    ]

    # The directory containing the files that were analyzed. We can search in here for the files,
    # even if the path doesn't match exactly. This is needed because we may have performed
    # the analysis on a different computer.
    SOURCE_DIR = "D:\\Recording\\Samples\\freesound\\creative_commons_0\\granulation"
    
    # The database
    DB_FILE = "D:\\Source\\grain_processor\\data\\grains.sqlite3"
    SELECT = [
        """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness BETWEEN 0.01 AND 0.1) 
            AND (length = 8192)
            AND (spectral_roll_off_75 BETWEEN 100 AND 500)
            AND (energy > 0.2);
        """,
        """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness BETWEEN 0.1 AND 0.2) 
            AND (length = 8192)
            AND (spectral_roll_off_75 BETWEEN 100 AND 500)
            AND (energy BETWEEN 0.1 AND 0.2);
        """,
        """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness BETWEEN 0.2 AND 0.3) 
            AND (length = 8192)
            AND (spectral_roll_off_75 BETWEEN 300 AND 1100)
            AND (energy > 0.1);
        """,
        """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness BETWEEN 0.0 AND 0.3) 
            AND (length = 8192)
            AND (spectral_roll_off_75 BETWEEN 1300 AND 1500)
            AND (energy > 0.1);
        """,
        """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness BETWEEN 0.3 AND 0.4) 
            AND (length = 8192)
            AND (spectral_roll_off_75 BETWEEN 1200 AND 1800)
            AND (energy > 0.1);
        """,
        """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness BETWEEN 0.1 AND 0.4) 
            AND (length = 8192)
            AND (spectral_roll_off_75 BETWEEN 1200 AND 2400)
            AND (energy > 0.1);
        """,
        """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness BETWEEN 0.5 AND 0.6) 
            AND (length = 8192)
            AND (spectral_roll_off_75 BETWEEN 1000 AND 3000)
            AND (energy > 0.1);
        """,
        """
        SELECT * 
        FROM grains 
        WHERE 
            (spectral_flatness BETWEEN 0.7 AND 1.0)
            AND (length = 8192);
        """
    ]

    # Retrieve grain metadata and grains
    db, cursor = grain_sql.connect_to_db(DB_FILE)
    grain_categories = []
    for i, select in enumerate(SELECT):
        category = grain_sql.realize_grains(cursor, select, SOURCE_DIR)
        if len(category) == 0:
            raise Exception(f"No grains found for index {i}.")
        grain_categories.append(category)
    db.close()

    # Generate candidate audio
    NUM_AUDIO_CANDIDATES = 1
    NUM_CHANNELS = 2
    for i in range(NUM_AUDIO_CANDIDATES):
        print(f"Generating audio candidate {i+1}...")
        # Each chunk will use 10 different grains
        NUM_UNIQUE_GRAINS = 20

        # Assemble the unique grain lists. There will be N lists, one for each SELECT statement.
        unique_grain_lists = []
        for j, category in enumerate(grain_categories):
            grain_list = []
            # select NUM unique grains
            for num in range(NUM_UNIQUE_GRAINS):
                grain_list.append(category[rng.randrange(0, len(category))])
            unique_grain_lists.append(grain_list)
        
        # Repeat the chunks to make longer audio
        repeated_grain_lists = []
        for j, unique_grain_list in enumerate(unique_grain_lists):
            repeated_grain_list = grain_assembler.assemble_repeat(unique_grain_list, 100, -8150, -18.0, effect_chain, None)

            # mess with channel indices         
            for k in range(0, len(repeated_grain_list)):
                repeated_grain_list[k]["channel"] = (k + 1) % NUM_CHANNELS
            
            repeated_grain_lists.append(repeated_grain_list)
        
        # Merge the grains into their final positions in an audio array
        grains = repeated_grain_lists[0][:len(repeated_grain_lists[0]) // 2]
        for j in range(1, len(repeated_grain_lists)):
            grains += grain_assembler.interpolate(repeated_grain_lists[j-1][len(repeated_grain_lists[j-1])//2:], repeated_grain_lists[j][:len(repeated_grain_lists[j])//2])
        grains += repeated_grain_lists[-1][len(repeated_grain_lists[-1])//2:]
        
        grain_assembler.calculate_grain_positions(grains)
        grain_audio = grain_assembler.merge(grains, NUM_CHANNELS, np.hanning)
        # grain_audio = operations.force_equal_energy(grain_audio, -12)
        
        # Apply final effects to the assembled audio
        lpf = signal.butter(2, 500, btype="lowpass", output="sos", fs=44100)
        hpf = signal.butter(8, 100, btype="highpass", output="sos", fs=44100)
        grain_audio = signal.sosfilt(lpf, grain_audio)
        grain_audio = signal.sosfilt(hpf, grain_audio)
        grain_audio = operations.fade_in(grain_audio, "hanning", 22050)
        grain_audio = operations.fade_out(grain_audio, "hanning", 22050)
        grain_audio = grain_audio * 0.2
        grain_audio = operations.adjust_level(grain_audio, -12)

        # Write the audio
        audio = audiofile.AudioFile(sample_rate=44100, bits_per_sample=24, num_channels=NUM_CHANNELS)
        audio.samples = grain_audio
        audiofile.write_with_pedalboard(audio, f"D:\\Recording\\temp9_{i+1}.wav")

        print("Done.")
