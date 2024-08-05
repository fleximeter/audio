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
import os
import platform
import multiprocessing as mp
from datetime import datetime


MAC = "/Users/jmartin50/recording"
ARGON = "/Users/jmartin50/recording"
PC = "D:\\recording"
SYSTEM = platform.system()

if SYSTEM == "Darwin":
    SOURCE_DIRS = os.path.join(MAC, "samples/granulation")
    OUT = os.path.join(MAC, "out")
    DB = os.path.join(MAC, "grains.sqlite3")
    
elif SYSTEM == "Linux":
    SOURCE_DIRS = [os.path.join(ARGON, "samples/granulation"), os.path.join("/old_Users/jmartin50/recording", "samples/granulation")]
    OUT = os.path.join(ARGON, "out")
    DB = os.path.join(ARGON, "grains.sqlite3")

else:
    SOURCE_DIRS = os.path.join(PC, "samples\\granulation")
    OUT = os.path.join(PC, "out")
    DB = os.path.join(PC, "grains.sqlite3")

print(f"Out directory: {OUT}\nSource directory: {SOURCE_DIRS}\nDatabase: {DB}")



def force_equal_energy(audio: np.ndarray, dbfs: float = -6.0, window_size: int = 8192, max_scalar: float = 1e6):
    """
    Forces equal energy on a mono signal over time. For example, if a signal initially has high energy, 
    and gets less energetic, this will adjust the energy level so that it does not decrease.
    Better results come with using a larger window size, so the energy changes more gradually.
    :param audio: The array of audio samples
    :param dbfs: The target level of the entire signal, in dbfs
    :param window_size: The window size to consider when detecting RMS energy
    :param max_scalar: The maximum scalar to use in level adjustment. This is necessary
    because some audio may have an extremely low maximum level, and the scalar required
    to bring it up to the right level could result in computation problems. The computed
    level scalar used in this function will be -max_scalar <= level <= max_scalar.
    :return: An adjusted version of the signal
    """
    # i: cython.int
    # j: cython.int
    # k: cython.int
    # idx: cython.int
    # frame_idx: cython.int
    if audio.ndim == 1:
        raise Exception("The audio array must have two dimensions.")
    num_channels = audio.shape[0]
    output_audio_arr = np.empty(audio.shape)  # the new array we'll be returning
    target_level = 10 ** (dbfs / 20)  # the target level, in float rather than dbfs
    num_frames = int(np.ceil(audio.shape[-1] / window_size))  # the number of frames that we'll be analyzing
    energy_levels = np.empty((num_channels, num_frames + 2))  # the energy level for each frame
    
    # find the energy levels
    for i in range(num_channels):
        idx = 1
        for j in range(0, audio.shape[-1], window_size):
            energy_levels[i, idx] = np.sqrt(np.average(np.square(audio[i, j:j+window_size])))
            idx += 1
        energy_levels[i, 0] = energy_levels[i, 1]
        energy_levels[i, -1] = energy_levels[i, -2]

    # do the first half frame
    for i in range(num_channels):
        scalar = target_level / energy_levels[i, 0]
        scalar = max(-max_scalar, scalar)
        scalar = min(max_scalar, scalar)
        for j in range(0, window_size // 2):
            output_audio_arr[i, j] = audio[i, j] * scalar
    
    # do adjacent half frames from 1 and 2, 2 and 3, etc.
    for i in range(num_channels):
        frame_idx = 1
        for frame_start_idx in range(window_size // 2, audio.shape[-1], window_size):
            slope = (energy_levels[i, frame_idx + 1] - energy_levels[i, frame_idx]) / window_size
            y_int = energy_levels[i, frame_idx]
            for sample_idx in range(frame_start_idx, min(frame_start_idx + window_size, audio.shape[-1])):
                f = slope * (sample_idx - frame_start_idx) + y_int
                scalar = 1/f
                scalar = max(-max_scalar, scalar)
                scalar = min(max_scalar, scalar)
                output_audio_arr[i, sample_idx] = audio[i, sample_idx] * scalar
            frame_idx += 1

    audio_max = np.max(np.abs(output_audio_arr))
    scalar = target_level / audio_max
    scalar = max(-max_scalar, scalar)
    scalar = min(max_scalar, scalar)
    return output_audio_arr * scalar
    

def render(grain_entry_categories, num_unique_grains_per_section, num_repetitions, overlap_num, num_channels, source_dirs, out_dir, name):
    """
    Renders an audio file
    :param grain_entry_categories: A list of grain record lists
    :param num_unique: The number of unique grains to use for each category
    :param num_channels: The number of channels in the output audio file
    :param source_dirs: The location(s) of the audio files
    :param out_dir: The output directory
    :param name: The output file name
    """
    DB = -6
    rng = random.Random()
    rng.seed()
    
    # Assemble the unique grain lists. There will be N lists, one for each SELECT statement.
    unique_grain_lists = []
    for j, entry_category in enumerate(grain_entry_categories):
        grain_list = []
        # select NUM unique grains
        for _ in range(num_unique_grains_per_section):
            idx = rng.randrange(0, len(entry_category))
            if "church-bell" not in entry_category[idx]["file"]:
                grain = entry_category[idx]
                grain["distance_between_grains"] = overlap_num
                grain["channel"] = 0
                grain_list.append(grain)
        # print(f"{len(grain_list)} grains added to the list")
        unique_grain_lists.append(grain_list)
    
    grains = []
    for l in unique_grain_lists:
        bigger_list = grain_assembler.assemble_repeat(l, num_repetitions, overlap_num)
        grains += bigger_list

    # Repeat the chunks to make longer audio
    
    grain_assembler.calculate_grain_positions(grains)
    grain_sql.realize_grains(grains, source_dirs)
    for i in range(len(grains)):
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], DB)
        
    grain_audio = grain_assembler.merge(grains, num_channels, np.hanning)
    
    grain_audio = force_equal_energy(grain_audio, -3, 22000)
    
    # # print("Ready to apply effects")

    # Apply final effects to the assembled audio
    lpf = signal.butter(2, 500, btype="lowpass", output="sos", fs=44100)
    hpf = signal.butter(8, 100, btype="highpass", output="sos", fs=44100)
    grain_audio = signal.sosfilt(lpf, grain_audio)
    grain_audio = signal.sosfilt(hpf, grain_audio)
    grain_audio = operations.fade_in(grain_audio, "hanning", 22050)
    grain_audio = operations.fade_out(grain_audio, "hanning", 22050)
    grain_audio = operations.adjust_level(grain_audio, -12)

    # print("Ready to write audio")

    # Write the audio
    audio = audiofile.AudioFile(sample_rate=44100, bits_per_sample=24, num_channels=num_channels)
    audio.samples = grain_audio
    path = os.path.join(out_dir, name)
    print(f"Writing file {path} with {audio.samples.shape[-1]} samples")
    audiofile.write_with_pedalboard(audio, os.path.join(out_dir, "out1.wav"))
    # print("Done.")


if __name__ == "__main__":
    LENGTH = 8192
    SELECT = [
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200)
            AND (energy > 0.05)
            AND (frequency IS NULL);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.1 AND 0.3) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 300)
            AND (energy > 0.05);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200)
            AND (energy > 0.05)
            AND (frequency IS NULL);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.1 AND 0.5) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 400)
            AND (energy > 0.05);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 75 AND 500)
            AND (energy > 0.05)
            AND (frequency IS NULL);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.1 AND 0.2) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200)
            AND (energy > 0.05);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 75 AND 600)
            AND (energy > 0.05)
            AND (frequency IS NULL);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.2 AND 0.8) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 500)
            AND (energy > 0.05);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 50 AND 800)
            AND (energy > 0.05)
            AND (frequency IS NULL);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.5 AND 1.0) 
            AND (spectral_roll_off_75 BETWEEN 100 AND 200)
            AND (energy > 0.05);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 50 AND 900)
            AND (energy > 0.05)
            AND (frequency IS NULL);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.3 AND 0.7) 
            AND (spectral_roll_off_75 BETWEEN 50 AND 1000)
            AND (energy > 0.05);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.00 AND 0.05) 
            AND (spectral_roll_off_75 BETWEEN 50 AND 1100)
            AND (energy > 0.05)
            AND (frequency IS NULL);""",
        f"""SELECT * FROM grains 
        WHERE (length = {LENGTH})
            AND (spectral_flatness BETWEEN 0.1 AND 0.4) 
            AND (spectral_roll_off_75 BETWEEN 20 AND 1400)
            AND (energy > 0.05);""",
    ]

    # EPSILON = 0.2
    # SELECT = [
    #     f"""SELECT * FROM grains 
    #         WHERE (length = 8192)
    #         AND (spectral_flatness BETWEEN 0.00 AND 0.05)
    #         AND (midi BETWEEN {69-EPSILON} AND {69 + EPSILON});""",
    #     f"""SELECT * FROM grains 
    #         WHERE (length = 8192)
    #         AND (spectral_flatness BETWEEN 0.00 AND 0.05)
    #         AND (midi BETWEEN {62-EPSILON} AND {62 + EPSILON});""",
    #     f"""SELECT * FROM grains 
    #         WHERE (length = 8192)
    #         AND (spectral_flatness BETWEEN 0.00 AND 0.05)
    #         AND (midi BETWEEN {64-EPSILON} AND {64 + EPSILON});""",
    #     f"""SELECT * FROM grains 
    #         WHERE (length = 8192)
    #         AND (spectral_flatness BETWEEN 0.00 AND 0.05)
    #         AND (midi BETWEEN {67-EPSILON} AND {67 + EPSILON});""",
    #     ]

    print("Retrieving grains...")
    # Retrieve grain metadata and grains
    db, cursor = grain_sql.connect_to_db(DB)
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
    start = datetime.now()
    print("Found grains")

    # Generate candidate audio
    NUM_AUDIO_CANDIDATES = 5
    NUM_CHANNELS = 2
    NUM_UNIQUE_GRAINS = 10
    render(grain_entry_categories, NUM_UNIQUE_GRAINS, 200, -8100, NUM_CHANNELS, SOURCE_DIRS, OUT, "out_1.wav")
    # processes = [mp.Process(target=render, args=(grain_entry_categories, NUM_UNIQUE_GRAINS, 800, -4050, NUM_CHANNELS, SOURCE_DIRS, OUT, f"out_{i+1}.wav")) for i in range(NUM_AUDIO_CANDIDATES)]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()
    duration = datetime.now() - start
    print("Elapsed time: {}:{:2}".format(duration.seconds // 60, duration.seconds % 60))
    