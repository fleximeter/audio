"""
File: grain_db.py

Description: Extracts grains from audio files, and records the frames and an analysis
"""

import os
import audiopython.audiofile as audiofile
import audiopython.granulator as granulator
import audiopython.sampler as sampler
import audiopython.analysis as analysis
import audiopython.operations as operations
import numpy as np
import random
import grain_sql


def analyze_grains(grains):
    """
    Analyzes grains and prepares for storage in the database
    :param grains: The grains to analyze
    :return: The analysis of the grains
    """
    analyses = []
    for grain in grains:
        a = audiofile.read(grain[1])
        g = granulator.extract_grain(np.reshape(a.samples, (a.samples.size)), grain[2], grain[3] - grain[2], max_window_size=512)
        ax = analysis.analyzer(g, a.sample_rate)
        analyses.append((
            grain[0],
            ax["dbfs"],
            ax["energy"],
            ax["pitch"],
            ax["midi"],
            ax["spectral_centroid"],
            ax["spectral_entropy"],
            ax["spectral_flatness"],
            ax["spectral_slope"][0],
            ax["spectral_slope"][1],
            ax["spectral_variance"],
            ax["spectral_skewness"],
            ax["spectral_kurtosis"],
            ax["spectral_roll_off_0.5"],
            ax["spectral_roll_off_0.75"],
            ax["spectral_roll_off_0.9"],
            ax["spectral_roll_off_0.95"],
            ax["zero_crossing_rate"],
            ))
    return analyses


def extract_random_grains(directory, n, size, dbfs_floor, instrument_family, instrument_name):
    """
    Extracts grains from all samples in a given directory
    :param directory: The directory with the samples
    :param n: The number of grains to extract
    :param size: The size in frames of each grain
    :param dbfs_floor: The lowest peak dBFS allowed for a grain
    :param instrument_family: The instrument family name
    :param instrument_name: The instrument name
    :return: A list of grains
    """
    grains = []
    MAX_JUMP = size * 10  # The maximum number of frames to jump forward randomly
    MIN_JUMP = 512  # The minimum number of frames to jump forward randomly

    # Load samples
    samples = sampler.load_samples(directory)
    print(f"Extracting grains from {directory}...")
    for sample in samples:
        sample.instrument_family = instrument_family.lower()
        sample.instrument_name = instrument_name.lower()
        i = 0
        start_pos = 0
        print(f"file: {os.path.split(sample.path)[-1]}: ", end="")

        # Try to extract n grains, but we cannot extract beyond the end of the sample.
        # We extract grains moving from the beginning to the end of the file.
        while i < n and start_pos + size < sample.frames:
            frame1 = start_pos + random.randrange(MIN_JUMP, MAX_JUMP)
            frame2 = frame1 + size
            start_pos = frame1
            if frame2 < sample.frames:
                # The grain we extracted
                grain = sample.samples[frame1:frame2]
                dbfs = operations.dbfs_max_local(grain, 1, 1)

                # If the grain is valid
                if dbfs >= dbfs_floor:
                    i += 1
                    grains.append((
                        sample.path,
                        frame1,
                        frame2,
                        sample.sample_rate,
                        (frame2 - frame1) / sample.sample_rate,
                        None,
                        sample.midi,
                        sample.dynamic_name,
                        sample.instrument_name,
                        sample.instrument_family,
                        sample.string_name
                    ))
        print(f"{i} grains extracted")
    
    return grains


def extract_random_grains_iowa(directory, n, size, dbfs_floor, instrument_family):
    """
    Extracts grains from all samples in a given directory. This function is customized
    for working with Iowa EMS samples, which have file names that have information
    that needs to be extracted and stored in the grain record.

    :param directory: The directory with the samples
    :param n: The number of grains to extract
    :param size: The size in frames of each grain
    :param dbfs_floor: The lowest peak dBFS allowed for a grain
    :param instrument_family: The instrument family name
    :return: A list of grains
    """
    grains = []
    MAX_JUMP = size + 2048  # The maximum number of frames to jump forward randomly
    MIN_JUMP = 512  # The minimum number of frames to jump forward randomly

    # Load samples
    samples = sampler.load_samples(directory, iowa=True)
    print(f"Extracting grains from {directory}...")
    for sample in samples:
        sample.instrument_family = instrument_family
        sample.instrument_name = sample.instrument_name.lower()
        i = 0
        start_pos = 0
        print(f"file: {os.path.split(sample.path)[-1]}: ", end="")

        # Try to extract n grains, but we cannot extract beyond the end of the sample.
        # We extract grains moving from the beginning to the end of the file.
        while i < n and start_pos + size < sample.frames:
            frame1 = start_pos + random.randrange(MIN_JUMP, MAX_JUMP)
            frame2 = frame1 + size
            start_pos = frame1
            if frame2 < sample.frames:
                # The grain we extracted
                grain = sample.samples[frame1:frame2]
                dbfs = operations.dbfs_max_local(grain, 1, 1)

                # If the grain is valid
                if dbfs >= dbfs_floor:
                    i += 1
                    grains.append((
                        sample.path,
                        frame1,
                        frame2,
                        sample.sample_rate,
                        (frame2 - frame1) / sample.sample_rate,
                        operations.midicps(sample.midi),
                        sample.midi,
                        sample.dynamic_name,
                        sample.instrument_name,
                        sample.instrument_family,
                        sample.string_name
                    ))
        print(f"{i} grains extracted")
    
    return grains


if __name__ == "__main__":
    random.seed()
    DB_FILE = "grains.sqlite3"
    DB_SCHEMA = "schema.sql"
    db, cursor = grain_sql.connect_to_db(DB_FILE, DB_SCHEMA)
    # SAMPLE_DIR = "D:\\Recording\\Samples\\Iowa\\Viola.arco.mono.2444.1\\samples"
    SAMPLE_DIR = "D:\\Recording\\Samples\\freesound\\creative_commons_0\\wind_chimes\\eq"
    # grains = extract_random_grains(SAMPLE_DIR, 200, 8192, -30, "bell", "wind chimes")
    # grain_sql.store_grains(grains, db, cursor)
    a = grain_sql.retrieve_grains(cursor, False)
    ax = analyze_grains(a)
    grain_sql.store_analyses(ax, db, cursor)
    db.close()
