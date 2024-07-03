"""
File: grain_assembler.py

Description: Assembles grains from a database based on specified parameters
"""

import grain_sql
import audiopython.audiofile as audiofile
import audiopython.granulator as granulator
import audiopython.sampler as sampler
import audiopython.analysis as analysis
import audiopython.operations as operations
import numpy as np
import random

FIELDS = [
    "id", "file", "start_frame", "end_frame", "sample_rate", "grain_duration",
    "frequency", "midi", "spectral_centroid", "spectral_entropy", "spectral_flatness",
    "spectral_kurtosis", "spectral_roll_off_50", "spectral_roll_off_75",
    "spectral_roll_off_90", "spectral_roll_off_95", "spectral_skewness", "spectral_slope",
    "spectral_slope_0_1_khz", "spectral_slope_1_5_khz", "spectral_slope_0_5_khz",
    "spectral_variance"
]


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


def assemble(grains, feature, line, interval):
    """
    Assembles grains given specified parameters
    :param grains: A list of grain dictionaries to choose from
    :param feature: The string name of the audio feature to use
    :param line: A line of numbers. Grains will be matched to each number.
    :param interval: The interval between each grain, in frames
    """


def realize_grains(cursor, sql):
    """
    Retrieves grains from the database and extracts the corresponding grains.
    :param sql: The SQL to use
    :return: A list of audio grain dictionaries
    """
    grains1 = cursor.execute(sql)
    audio = {}  # Holds the unique audio files that we are extracting grains from
    grains2 = []  # A list of grain dictionaries
    for grain_tup in grains1:
        grain = {FIELDS[i]: grain_tup[i] for i in range(len(grain_tup))}
        if grain["file"] not in audio:
            audio_data = audiofile.read(grain["file"])
            audio[grain["file"]] = audio_data.samples[0]
        grains2.append(grain.update({"grain": audio[grain["file"]][grain["start_frame"]:grain["end_frame"]]}))
    return grains2    


if __name__ == "__main__":
    random.seed()
    DB_FILE = "D:\\Source\\grain_processor\\data\\grains.sqlite3"
    db, cursor = grain_sql.connect_to_db(DB_FILE)
    grains1 = realize_grains(cursor, "SELECT * FROM grains WHERE frequency BETWEEN 100.0 AND 150.0;")
    