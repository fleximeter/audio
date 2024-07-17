"""
File: grain_assembler.py

Description: Assembles grains from a database based on specified parameters
"""

import grain_sql
import aus.audiofile as audiofile
import aus.granulator as granulator
import aus.sampler as sampler
import aus.analysis as analysis
import aus.operations as operations
import numpy as np
import os
import random
import scipy.signal as signal
from effects import *


FIELDS = [
    "id", "file", "start_frame", "end_frame", "length", "sample_rate", "grain_duration",
    "frequency", "midi", "energy", "spectral_centroid", "spectral_entropy", "spectral_flatness",
    "spectral_kurtosis", "spectral_roll_off_50", "spectral_roll_off_75",
    "spectral_roll_off_90", "spectral_roll_off_95", "spectral_skewness", "spectral_slope",
    "spectral_slope_0_1_khz", "spectral_slope_1_5_khz", "spectral_slope_0_5_khz",
    "spectral_variance"
]


def find_path(database_path, parent_directory) -> str:
    """
    Resolves a database path to a path on the local machine, using a parent directory to search.
    Searches the parent directory for a file that matches the file name in the database.
    Note:
    - The file name must match exactly the file name on this computer, including file extension and case.
    - If there are multiple files located somewhere under the provided parent directory, this function might
      not find the right file. Don't have duplicate file names in the database.
    :param database_path: The path of the file in the database
    :param parent_directory: The directory containing the file
    :return: The actual file path on this machine
    """
    file_name = os.path.split(database_path)[-1]
    for path, _, files in os.walk(parent_directory):
        for file in files:
            if file_name in file:
                return os.path.join(path, file)
    return ""


def line(start: float, end: float, length: int):
    """
    Generates an array with linear content
    :param start: The start value
    :param end: The end value
    :param length: The length of the array
    :return: A NumPy array of the provided specifications
    """
    slope = (end - start) / length
    line_arr = np.zeros((length))
    for i in range(length):
        line_arr[i] = slope * i + start
    return line_arr


def assemble(grains, features, interval, max_db=-18.0):
    """
    Assembles grains given specified parameters. Applies a window norm to the final assembled grains.
    :param grains: A list of grain dictionaries to choose from
    :param feature: The string name of the audio feature to use
    :param line: A line of numbers. Grains will be matched to each number.
    :param interval: The interval between each grain, in frames
    """
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

    # Organize the grains
    for feature in features:
        grains = sorted(grains, key=lambda x: x[feature])

    # Create the window and initialize the window norm
    window = np.hanning(grains[0]["length"])
    window_norm = window.copy()

    # Apply effects
    for effect in effect_chain:
        grains[0]["grain"] = effect(grains[0]["grain"])
    grains[0]["grain"] = effect_cycle[0](grains[0]["grain"])

    # Add the grain to the audio sequence
    grains[0]["grain"] = operations.adjust_level(grains[0]["grain"], max_db)
    final_audio = grains[0]["grain"] * window
    last_grain_onset = 0
    for i in range(1, len(grains)):
        # Apply effects
        window = np.hanning(grains[i]["length"])
        for effect in effect_chain:
            grains[i]["grain"] = effect(grains[i]["grain"])
        grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        grains[i]["grain"] *= window

        # Add the grain to the audio sequence and update the window norm
        overlap_num = final_audio.shape[-1] - last_grain_onset - interval
        new_grain_onset = last_grain_onset + interval
        final_audio = np.hstack((final_audio[:new_grain_onset], final_audio[new_grain_onset:] + grains[i]["grain"][:overlap_num], grains[i]["grain"][overlap_num:]))
        window_norm = np.hstack((window_norm[:new_grain_onset], window_norm[new_grain_onset:] + window[:overlap_num], window[overlap_num:]))
        last_grain_onset = new_grain_onset
    
    final_audio /= window_norm
    final_audio = np.nan_to_num(final_audio)
    return final_audio


def realize_grains(cursor, sql, source_dir):
    """
    Retrieves grains from the database and extracts the corresponding grains.
    :param cursor: A database cursor
    :param sql: The SQL to use
    :param source_dir: The directory that contains the audio files to extract grains from.
    This is needed because this might not be the directory the audio files were contained
    in when the granulation analysis was performed.
    :return: A list of audio grain dictionaries
    """
    grains1 = cursor.execute(sql)
    audio = {}  # Holds the unique audio files that we are extracting grains from
    grains2 = []  # A list of grain dictionaries
    for grain_tup in grains1:
        grain = {FIELDS[i]: grain_tup[i] for i in range(len(grain_tup))}
        if grain["file"] not in audio:
            print(grain["file"])
            audio_data = audiofile.read(find_path(grain["file"], source_dir))
            audio[grain["file"]] = audio_data.samples[0]
        grain["spectral_roll_off_50"] = round(grain["spectral_roll_off_50"], 2)
        grain["spectral_centroid"] = round(grain["spectral_centroid"], -1)
        grain.update({"grain": audio[grain["file"]][grain["start_frame"]:grain["end_frame"]]})
        grains2.append(grain)
    return grains2


if __name__ == "__main__":
    random.seed()
    
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
            (spectral_flatness < 0.2) 
            AND (length = 8192)
            AND (spectral_roll_off_75 < 1000)
            AND (energy > 0.2);
    """
    db, cursor = grain_sql.connect_to_db(DB_FILE)
    grains1 = realize_grains(cursor, SELECT, SOURCE_DIR)
    audio = audiofile.AudioFile(sample_rate=44100, bits_per_sample=24, num_channels=1)
    samples = assemble(grains1, ["spectral_roll_off_50", "spectral_slope_0_1_khz"], 3000)
    
    lpf = signal.butter(2, 500, btype="lowpass", output="sos", fs=44100)
    hpf = signal.butter(8, 100, btype="highpass", output="sos", fs=44100)
    samples = signal.sosfilt(lpf, samples)
    samples = signal.sosfilt(hpf, samples)
    samples = operations.fade_in(samples, "hanning", 22050)
    samples = operations.fade_out(samples, "hanning", 22050)
    samples = operations.adjust_level(samples, -12)
    audio.samples = samples

    audiofile.write_with_pedalboard(audio, "D:\\Recording\\temp.wav")
    db.close()
