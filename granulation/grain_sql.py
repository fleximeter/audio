"""
File: grain_sql.py

Description: Works with SQL database for granulation
"""

import sqlite3
import aus.audiofile as audiofile
import numpy as np
import os


FIELDS = [
    "id", "file", "start_frame", "end_frame", "length", "sample_rate", "grain_duration",
    "frequency", "midi", "energy", "spectral_centroid", "spectral_entropy", "spectral_flatness",
    "spectral_kurtosis", "spectral_roll_off_50", "spectral_roll_off_75",
    "spectral_roll_off_90", "spectral_roll_off_95", "spectral_skewness", "spectral_slope",
    "spectral_slope_0_1_khz", "spectral_slope_1_5_khz", "spectral_slope_0_5_khz",
    "spectral_variance"
]


def connect_to_db(path):
    """
    Connects to a SQLite database
    :param path: The path to the SQLite database
    :return: Returns the database connection and a cursor for SQL script execution
    NOTE: You will need to manually close the database connection that is returned from this function!
    """
    db = sqlite3.connect(path)
    cursor = db.cursor()
    return db, cursor


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
            # print(grain["file"])
            audio_data = audiofile.read(find_path(grain["file"], source_dir))
            audio[grain["file"]] = audio_data.samples[0]
        grain["spectral_roll_off_50"] = round(grain["spectral_roll_off_50"], 2)
        grain["spectral_centroid"] = round(grain["spectral_centroid"], -1)
        grain.update({"grain": audio[grain["file"]][grain["start_frame"]:grain["end_frame"]]})
        # weed out grains with bad values
        if not (np.isnan(grain["grain"]).any() or np.isinf(grain["grain"]).any() or np.isneginf(grain["grain"]).any()):
            grains2.append(grain)
    return grains2


def retrieve_grains(cursor):
    """
    Retrieves grains from the database
    :param cursor: The cursor for executing SQL
    :param analyzed: Whether to retrieve only grains that have been analyzed or have not been analyzed. 
    If None, will retrive all grains. If True, will retrieve only analyzed grains. 
    If False, will retrieve only unanalyzed grains.
    :return: The grains
    """
    SQL = """
        SELECT *
        FROM grains;
        """
    return cursor.execute(SQL)


def store_grains(grains, db, cursor):
    """
    Stores grains in the database
    :param grains: A list of grain dictionaries
    :param db: A connection to a SQLite database
    :param cursor: The cursor for executing SQL
    """
    SQL = "INSERT INTO grains VALUES(NULL, " + "?, " * 20 + "?)"
    cursor.executemany(SQL, grains)
    db.commit()
