"""
File: grain_sql.py

Description: Works with SQL database for granulation
"""


import os
import sqlite3
import audiopython.audiofile as audiofile
import audiopython.granulator as granulator
import audiopython.sampler as sampler
import audiopython.analysis as analysis
import audiopython.operations as operations
import numpy as np


def connect_to_db(path, schema_path):
    """
    Connects to a SQLite database and populates the tables if necessary

    :param path: The path to the SQLite database
    :param schema_path: The path to the SQL schema file if the database does not exist
    :return: Returns the database connection and a cursor for SQL script execution
    
    NOTE: You will need to manually close the database connection that is returned from this function!
    """
    if os.path.exists(path):
        db = sqlite3.connect(path)
        cursor = db.cursor()
    else:
        db = sqlite3.connect(path)
        cursor = db.cursor()
        with open(schema_path, 'r') as schema:
            schema_text = schema.read()
            cursor.executescript(schema_text)
            db.commit()
    return db, cursor


def retrieve_grains(cursor, analyzed=None):
    """
    Retrieves grains from the database
    :param cursor: The cursor for executing SQL
    :param analyzed: Whether to retrieve only grains that have been analyzed or have not been analyzed. 
    If None, will retrive all grains. If True, will retrieve only analyzed grains. 
    If False, will retrieve only unanalyzed grains.
    :return: The grains
    """
    if analyzed is None:
        SQL = """
            SELECT *
            FROM grains;
            """
    elif analyzed == True:
        SQL = """
            SELECT grains.*
            FROM grains
            LEFT JOIN analysis ON grains.id = analysis.grain_id;
            """
    else:
        SQL = """
            SELECT grains.*
            FROM grains
            LEFT JOIN analysis ON grains.id = analysis.grain_id 
            WHERE analysis.grain_id IS NULL;
            """
    return cursor.execute(SQL)


def store_analyses(analyses, db, cursor):
    """
    Stores analyses in the database
    :param analyses: A list of grain analyses
    :param db: A connection to a SQLite database
    :param cursor: The cursor for executing SQL
    """
    SQL = "INSERT INTO analysis VALUES(NULL, " + "?, " * 17 + "?)"
    cursor.executemany(SQL, analyses)
    db.commit()


def store_grains(grains, db, cursor):
    """
    Stores grains in the database
    :param grains: A list of grain dictionaries
    :param db: A connection to a SQLite database
    :param cursor: The cursor for executing SQL
    """
    SQL = "INSERT INTO grains VALUES(NULL, " + "?, " * 10 + "?)"
    cursor.executemany(SQL, grains)
    db.commit()