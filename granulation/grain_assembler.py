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


if __name__ == "__main__":
    random.seed()
    DB_FILE = "grains.sqlite3"
    DB_SCHEMA = "schema.sql"
    db, cursor = grain_sql.connect_to_db(DB_FILE, DB_SCHEMA)
