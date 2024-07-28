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
    SOURCE_DIR = "D:\\Recording\\Samples\\granulation"
    
    # The database
    DB_FILE = "D:\\Source\\grain_processor\\data\\grains.sqlite3"
    SELECT = [
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200)
            AND (frequency IS NULL);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.1 AND 0.3) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 300);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200)
            AND (frequency IS NULL);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.1 AND 0.5) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 400);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 75 AND 500)
            AND (frequency IS NULL);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.1 AND 0.2) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 75 AND 600)
            AND (frequency IS NULL);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.2 AND 0.8) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 500);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 50 AND 800)
            AND (frequency IS NULL);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.5 AND 1.0) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 50 AND 900)
            AND (frequency IS NULL);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.3 AND 0.7) 
            AND (spectral_roll_off_75 BETWEEN 50 AND 1000);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 50 AND 1100)
            AND (frequency IS NULL);""",
        """SELECT * FROM grains 
        WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.1 AND 0.4) 
            AND (spectral_roll_off_75 BETWEEN 20 AND 1400);""",
    ]

    EPSILON = 0.2
    SELECT = [
        f"""SELECT * FROM grains 
            WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05)
            AND (midi BETWEEN {69-EPSILON} AND {69 + EPSILON});""",
        f"""SELECT * FROM grains 
            WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05)
            AND (midi BETWEEN {62-EPSILON} AND {62 + EPSILON});""",
        f"""SELECT * FROM grains 
            WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05)
            AND (midi BETWEEN {64-EPSILON} AND {64 + EPSILON});""",
        f"""SELECT * FROM grains 
            WHERE (length = 8192)
            AND (spectral_flatness BETWEEN 0.00 AND 0.05)
            AND (midi BETWEEN {67-EPSILON} AND {67 + EPSILON});""",
        ]

    print("Retrieving grains...")
    # Retrieve grain metadata and grains
    db, cursor = grain_sql.connect_to_db(DB_FILE)
    grain_entry_categories = []
    for i, select in enumerate(SELECT):
        cursor.execute(select)
        records = cursor.fetchall()
        if len(records) == 0:
            raise Exception(f"No grains found for index {i}.")
        entry_category = []
        for record in records:
            entry_category.append({grain_sql.FIELDS[i]: record[i] for i in range(len(record))})
        grain_entry_categories.append(entry_category)
    db.close()

    # Generate candidate audio
    NUM_AUDIO_CANDIDATES = 5
    NUM_CHANNELS = 2
    NUM_UNIQUE_GRAINS = 10
    for i in range(NUM_AUDIO_CANDIDATES):
        print(f"Generating audio candidate {i+1}...")
        # Each chunk will use 10 different grains
        
        # Assemble the unique grain lists. There will be N lists, one for each SELECT statement.
        unique_grain_lists = []
        for j, entry_category in enumerate(grain_entry_categories):
            grain_list = []
            # select NUM unique grains
            for num in range(NUM_UNIQUE_GRAINS):
                idx = rng.randrange(0, len(entry_category))
                if "church-bell" not in entry_category[idx]["file"]:
                    grain_list.append(entry_category[idx])
            grain_list = grain_sql.realize_grains(grain_list, SOURCE_DIR)
            unique_grain_lists.append(grain_list)
        
        # Repeat the chunks to make longer audio
        repeated_grain_lists = []
        for j, unique_grain_list in enumerate(unique_grain_lists):
            repeated_grain_list = grain_assembler.assemble_repeat(unique_grain_list, 500, -8150, -18.0, effect_chain, None)
            grain_assembler.swap_nth_adjacent_pair(repeated_grain_list, 9)

            # mess with channel indices
            for k in range(0, len(repeated_grain_list)):
                repeated_grain_list[k]["channel"] = (k + 1) % NUM_CHANNELS
            
            repeated_grain_lists.append(repeated_grain_list)
        
        # Merge the grains into their final positions in an audio array
        # grains = []
        # for j in range(1, len(repeated_grain_lists)):
        #     grains += grain_assembler.interpolate(repeated_grain_lists[j-1][len(repeated_grain_lists[j-1])//2:], repeated_grain_lists[j][:len(repeated_grain_lists[j])//2])
        
        # grain_assembler.calculate_grain_positions(grains)
        # grain_audio = grain_assembler.merge(grains, NUM_CHANNELS, np.hanning)
        
        grains = []
        for j in range(1, len(repeated_grain_lists)):
            grain_assembler.calculate_grain_positions(repeated_grain_lists[j])
            grains.append(grain_assembler.merge(repeated_grain_lists[j], NUM_CHANNELS, np.hanning))
        
        # grain_audio = grains[0]
        # for grain in grains:
        #     grain_audio = np.hstack((grain_audio, np.zeros((2, 22050)), grain))
        grain_audio = grain_assembler.merge_crossfade(grains)
        grain_audio = operations.force_equal_energy(grain_audio, -3, 22050)
        
        # Apply final effects to the assembled audio
        lpf = signal.butter(2, 500, btype="lowpass", output="sos", fs=44100)
        hpf = signal.butter(8, 100, btype="highpass", output="sos", fs=44100)
        grain_audio = signal.sosfilt(lpf, grain_audio)
        grain_audio = signal.sosfilt(hpf, grain_audio)
        grain_audio = operations.fade_in(grain_audio, "hanning", 22050)
        grain_audio = operations.fade_out(grain_audio, "hanning", 22050)
        grain_audio = operations.adjust_level(grain_audio, -3)

        # Write the audio
        audio = audiofile.AudioFile(sample_rate=44100, bits_per_sample=24, num_channels=NUM_CHANNELS)
        audio.samples = grain_audio
        audiofile.write_with_pedalboard(audio, f"D:\\Recording\\temp9_{i+1}.wav")

        print("Done.")
