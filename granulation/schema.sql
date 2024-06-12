-- The grain table
CREATE TABLE grains (
    id INTEGER PRIMARY KEY,           -- ID column
    path TEXT NOT NULL,               -- The path of the file from which the grains are extracted
    frame1 INTEGER NOT NULL,          -- The starting frame of the grain
    frame2 INTEGER NOT NULL,          -- The ending frame of the grain
    sample_rate INTEGER NOT NULL,     -- The sample rate of the file
    grain_duration REAL NOT NULL,     -- The grain duration in seconds
    frequency REAL,                   -- The frequency (if the grain is pitched, otherwise NULL. This should be manually specified, not calculated with a pitch detection algorithm.)
    midi REAL,                        -- The MIDI note number (if the grain is pitched, otherwise NULL)
    dynamic TEXT,                     -- The dynamic
    instrument_name TEXT,             -- The instrument name (if an instrument produced the sound)
    instrument_family TEXT,           -- The instrument family (if an instrument produced the sound)
    string_name TEXT                  -- The string name (if the instrument is stringed)
);

-- The analysis table
CREATE TABLE analysis (
    id INTEGER PRIMARY KEY,           -- ID column
    grain_id INTEGER NOT NULL,        -- Grain ID column
    dbfs REAL,                        -- The max dBFS
    energy REAL,                      -- The energy of the grain
    pitch_estimation REAL,            -- The pitch estimation of the grain
    midi_estimation REAL,             -- The MIDI note estimation of the grain
    spectral_centroid REAL,           -- The spectral centroid
    spectral_entropy REAL,            -- The spectral entropy
    spectral_flatness REAL,           -- The spectral flatness
    spectral_slope REAL,              -- The spectral slope
    spectral_y_int REAL,              -- The spectral y intersection
    spectral_variance REAL,           -- The spectral variance
    spectral_skewness REAL,           -- The spectral skewness
    spectral_kurtosis REAL,           -- The spectral kurtosis
    spectral_roll_off_50 REAL,        -- The spectral roll off 50%
    spectral_roll_off_75 REAL,        -- The spectral roll off 75%
    spectral_roll_off_90 REAL,        -- The spectral roll off 90%
    spectral_roll_off_95 REAL,        -- The spectral roll off 95%
    zero_crossing_rate REAL           -- The zero crossing rate
);
