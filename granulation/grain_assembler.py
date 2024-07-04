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
import scipy.signal as signal

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


def assemble(grains, features, interval, max_db=-18.0):
    """
    Assembles grains given specified parameters. Applies a window norm to the final assembled grains.
    :param grains: A list of grain dictionaries to choose from
    :param feature: The string name of the audio feature to use
    :param line: A line of numbers. Grains will be matched to each number.
    :param interval: The interval between each grain, in frames
    """
    for feature in features:
        grains = sorted(grains, key=lambda x: x[feature])
    window = np.hanning(grains[0]["grain"].size)
    window_norm = window.copy()
    grains[0]["grain"] = operations.adjust_level(grains[0]["grain"], max_db)
    final_audio = grains[0]["grain"] * window
    for i in range(1, len(grains)):
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        grains[i]["grain"] *= window
        final_audio = np.hstack((final_audio[:-interval], final_audio[-interval:] + grains[i]["grain"][:interval], grains[i]["grain"][interval:]))
        window_norm = np.hstack((window_norm[:-interval], window_norm[-interval:] + window[:interval], window[interval:]))
    final_audio /= window_norm
    final_audio = np.nan_to_num(final_audio)
    return final_audio


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
        grain.update({"grain": audio[grain["file"]][grain["start_frame"]:grain["end_frame"]]})
        grains2.append(grain)
    return grains2


if __name__ == "__main__":
    random.seed()
    DB_FILE = "D:\\Source\\grain_processor\\data\\grains.sqlite3"
    db, cursor = grain_sql.connect_to_db(DB_FILE)
    grains1 = realize_grains(cursor, "SELECT * FROM grains WHERE frequency BETWEEN 100.0 AND 300.0;")
    audio = audiofile.AudioFile(sample_rate=44100, bits_per_sample=24, num_channels=1)
    samples = assemble(grains1, ["spectral_centroid", "spectral_slope"], 4000)

    lpf = signal.butter(2, 500, btype="lowpass", output="sos", fs=44100)
    hpf = signal.butter(8, 100, btype="highpass", output="sos", fs=44100)
    samples = signal.sosfilt(lpf, samples)
    samples = signal.sosfilt(hpf, samples)
    audio.samples = samples

    audiofile.write_with_pedalboard(audio, "D:\\Recording\\temp.wav")
