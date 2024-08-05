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
    # Need to compensate for os.path.split() not working properly on paths for other platform
    idx = len(database_path) - 1
    while idx >= 0:
        if database_path[idx] == "\\" or database_path[idx] == "/":
            break
        idx -= 1
    if idx >= 0:
        database_path = database_path[idx+1:]
    # print(f"Trying to find file {database_path}")
    if type(parent_directory) == list:
        for dir in parent_directory:
            for path, _, files in os.walk(dir):
                for file in files:
                    if database_path in file:
                        return os.path.join(path, file)
    elif type(parent_directory) == str:
        for path, _, files in os.walk(parent_directory):
            for file in files:
                if database_path in file:
                    return os.path.join(path, file)

    return ""


def read_grains_from_file(grain_entries: list, source_dir):
    """
    Extracts the corresponding grains from database records.
    :param grain_entries: The grain records to use
    :param source_dir: The directory that contains the audio files to extract grains from.
    This is needed because this might not be the directory the audio files were contained
    in when the granulation analysis was performed.
    """
    # Group the grains by source file
    grain_groups = {}
    for i, grain in enumerate(grain_entries):
        if grain["file"] not in grain_groups:
            grain_groups[grain["file"]] = []
        grain_groups[grain["file"]].append(i)
    
    for audio_file, grain_list in grain_groups.items():
        path = find_path(audio_file, source_dir)
        if not os.path.exists(path):
            print(f"Could not find path {path} for file {grain['file']}")
            print(f"The source directory was {source_dir}")
        else:
            audio = audiofile.read(path)
            for idx in grain_list:
                grain_entries[idx]["grain"] = audio.samples[0][grain_entries[idx]["start_frame"]:grain_entries[idx]["end_frame"]]
            del audio.samples
            del audio


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


def update_grain_root(cursor: sqlite3.Cursor, root_dir: str, root_dir_path: str):
    """
    Updates the path of all grains with `root_dir` in their paths.
    :param root_dir: The root directory
    :param root_dir_path: The new path to this root directory
    """
    SQL = "SELECT id, file FROM grains WHERE file LIKE ? OR file LIKE ?;"
    cursor.execute(SQL, (f"%{root_dir}/%", f"%{root_dir}\\%"))
    records = cursor.fetchall()
    newrecords = []
    for record in records:
        path = record[1].replace("\\", "/")
        path = path.split(f"{root_dir}/")[1]
        path = os.path.join(root_dir_path, path)
        newrecords.append((path, record[0]))
    print(f"Updating {len(records)} records")
    SQL = "UPDATE grains SET file = ? WHERE id = ?;"
    cursor.executemany(SQL, newrecords)


if __name__ == "__main__":
    DB = "D:\\grains.sqlite3"
    ROOT = "granulation_public_domain"
    NEWDIR = "D:/Recording/Samples/granulation"
    db, cursor = connect_to_db(DB)
    # update_grain_root(cursor, ROOT, NEWDIR)
    db.commit()
    db.close()
