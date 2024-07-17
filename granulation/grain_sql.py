"""
File: grain_sql.py

Description: Works with SQL database for granulation
"""

import sqlite3
import aus.audiofile as audiofile
import aus.granulator as granulator
import aus.sampler as sampler
import aus.analysis as analysis
import aus.operations as operations
import numpy as np


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
