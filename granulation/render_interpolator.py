"""
File: render_interpolator.py

This grain realizer produces chunks of grains with different characteristics 
and interpolates from chunk to chunk to make a final audio product.
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
    SOURCE_DIR = "D:\\Recording\\Samples\\granulation_public_domain"
    
    # The database
    DB_FILE = "D:\\Source\\grain_processor\\data\\grains.sqlite3"
    SELECT = [
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.1 AND 0.3) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 300);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.1 AND 0.5) 
            AND (spectral_roll_off_75 BETWEEN 200 AND 400);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 200 AND 500);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.1 AND 0.2) 
            AND (spectral_roll_off_75 BETWEEN 400 AND 6000);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 300 AND 600);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.2 AND 0.8) 
            AND (spectral_roll_off_75 BETWEEN 500 AND 700);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 500 AND 800);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.5 AND 1.0) 
            AND (spectral_roll_off_75 BETWEEN 700 AND 800);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 700 AND 900);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.3 AND 0.7) 
            AND (spectral_roll_off_75 BETWEEN 800 AND 900);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 800 AND 1000);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.1 AND 0.4) 
            AND (spectral_roll_off_75 BETWEEN 900 AND 1000);""",
    ]

    print("Retrieving grains...")
    # Retrieve grain metadata and grains
    db, cursor = grain_sql.connect_to_db(DB_FILE)
    grain_entry_categories = []
    for i, select in enumerate(SELECT):
        cursor.execute(select)
        entry_category = cursor.fetchall()
        if len(entry_category) == 0:
            raise Exception(f"No grains found for index {i}.")
        grain_entry_categories.append(entry_category)
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
        for j, entry_category in enumerate(grain_entry_categories):
            grain_list = []
            # select NUM unique grains
            for num in range(NUM_UNIQUE_GRAINS):
                grain_list.append(entry_category[rng.randrange(0, len(entry_category))])
            grain_list = grain_sql.realize_grains(grain_list, SOURCE_DIR)
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
        grain_audio = operations.adjust_level(grain_audio, -3)

        # Write the audio
        audio = audiofile.AudioFile(sample_rate=44100, bits_per_sample=24, num_channels=NUM_CHANNELS)
        audio.samples = grain_audio
        audiofile.write_with_pedalboard(audio, f"D:\\Recording\\temp9_{i+1}.wav")

        print("Done.")
