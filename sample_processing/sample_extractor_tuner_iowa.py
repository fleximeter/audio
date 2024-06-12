"""
File: sample_extractor_tuner_iowa.py
Author: Jeff Martin
Date: 12/22/23

This file loads all audio files with a directory and its subdirectories,
and extracts individual samples from them. It also tunes samples to the nearest MIDI note.
This program is customized for the samples recorded by the University of Iowa EMS.

The idea behind the sample extraction process is that you identify regions in the audio file
where the levels are consistently above a certain dBFS threshold.
"""

import audiopython.analysis as analysis
import audiopython.audiofile as audiofile
import audiopython.operations as operations
import audiopython.sampler as sampler
import multiprocessing as mp
import numpy as np
import os
import platform
import re
import scipy.signal


###################################################################################################
# Directory roots. We automatically detect if we're on Windows or Mac.
# NOTE: If you are someone other than the original author, running this code, you will need
# to set these paths manually. The only path that matters is the ROOT path if you are just
# using this on one computer.
###################################################################################################
WINROOT = "D:\\"
MACROOT = "/Volumes/AudioJeff"
PLATFORM = platform.platform()
ROOT = WINROOT
if re.search(r'macos', PLATFORM, re.IGNORECASE):
    ROOT = MACROOT

###################################################################################################
# !! THINGS THAT MUST BE MANUALLY SET EACH TIME YOU RUN THIS PROGRAM !!
###################################################################################################
DIR = os.path.join(ROOT, "Recording", "Samples", "Iowa", "Xylophone.rosewood")
DYNAMIC = "ff"  # This is something specific to Iowa samples

# 1. The minimum number of frames in a sample. The sample will actually be a bit longer; 
# among other things, it will include the POST_FRAMES_TO_INCLUDE.
MIN_SAMPLE_LENGTH = 11000

# 2. The level at which a sample begins or ends. When the levels rise above here, we have another sample.
SAMPLE_LEVEL_DBFS_DELIMITER = -36  

# 3. The number of frames that will be included at the end of the sample. This helps to catch the tail
# as it fades away. If this is too long, we might catch part of the next sample.
POST_FRAMES_TO_INCLUDE = 11000

# 4. If you want to automatically tune the sample, set this to True.
AUTOTUNE_SAMPLE = True


###################################################################################################
# THINGS YOU SHOULD GENERALLY LEAVE ALONE
###################################################################################################
CPU_COUNT = mp.cpu_count()
PEAK_DBFS_FOR_FINAL_SAMPLES = -12
SAMPLE_RATE = 44100
LOWCUT_FREQ = 55
LOWCUT = True

# The filter we use to remove DC bias and any annoying low frequency stuff. It is more than just a 
# DC bias filter because sometimes there is low frequency content we want to remove as well.
# If you are working with really low-frequency samples, you will need to lower the LOWCUT_FREQ constant.
filt = scipy.signal.butter(4, LOWCUT_FREQ, 'high', output='sos', fs=SAMPLE_RATE)


def extract_samples(audio_files, destination_directory):
    """
    Extracts samples from a list of provided audio files.
    :param audio_files: A list of audio file names
    :param destination_directory: The destination sample directory
    """
    for file in audio_files:
        short_name = re.sub(r'(\.wav$)|(\.aif+$)', '', os.path.split(file)[-1], re.IGNORECASE)
        
        # Read the audio file and force it to the right number of dimensions
        audio = audiofile.read(file)
        audio.samples = operations.mix_if_not_mono(audio.samples, 2)
        audio.num_channels = 1
        
        # Perform preprocessing
        if LOWCUT:
            audio.samples = scipy.signal.sosfilt(filt, audio.samples)
        
        # Extract the samples. You may need to tweak some settings here to optimize sample extraction.
        amplitude_regions = sampler.identify_amplitude_regions(audio=audio, level_delimiter=SAMPLE_LEVEL_DBFS_DELIMITER, num_consecutive=MIN_SAMPLE_LENGTH)
        samples = sampler.extract_samples(audio=audio, amplitude_regions=amplitude_regions, pre_frames_to_include=500, 
                                          post_frames_to_include=POST_FRAMES_TO_INCLUDE, pre_envelope_frames=500, post_envelope_frames=500)
        
        # Perform postprocessing, including scaling dynamic level and tuning
        for i, sample in enumerate(samples):
            sample.samples = operations.leak_dc_bias_averager(sample.samples)
            current_peak = np.max(np.abs(sample.samples))
            sample.samples *= 10 ** (PEAK_DBFS_FOR_FINAL_SAMPLES / 20) / current_peak
            if AUTOTUNE_SAMPLE:
                midi = analysis.midi_estimation_from_pitch(analysis.librosa_pitch_estimation(operations.mix_if_not_mono(
                    sample.samples
                    ), 44100, 27.5, 5000, 0.5))
                if not np.isnan(midi) and not np.isinf(midi) and not np.isneginf(midi):
                    sample.samples = operations.midi_tuner(sample.samples, midi, 1, 44100)
                    sample.num_frames = sample.samples.shape[-1]
                    midi = int(np.round(midi))
            audiofile.write_with_pedalboard(sample, os.path.join(destination_directory, f"{short_name}.{i+1}.wav"))



if __name__ == "__main__":
    print("Starting sample extractor...")
    destination_directory = os.path.join(DIR, "samples")
    os.makedirs(destination_directory, 511, True)

    files = audiofile.find_files(DIR)
    files2 = []
    # A basic file filter. We exclude samples that have already been created, because
    # they have "sample." in the file name. We also are targeting samples of a specific
    # dynamic level here.
    for file in files:
        if re.search(DYNAMIC, file, re.IGNORECASE) and not re.search(r'sample\.', file, re.IGNORECASE):
            files2.append(file)
    
    # Distribute the audio files among the different processes. This is a good way to do it
    # because we assume that some files will be harder to process, and those will probably
    # be adjacent to each other in the folder, so we don't want to take blocks of files;
    # we want to distribute them individually.
    file_groups = [[] for i in range(CPU_COUNT)]
    for i, file in enumerate(files2):
        file_groups[i % CPU_COUNT].append(file)

    # Start the processes
    processes = []
    for i in range(CPU_COUNT):
        processes.append(mp.Process(target=extract_samples, args=(file_groups[i], destination_directory)))
        processes[-1].start()
    
    # Collect the processes
    for i in range(CPU_COUNT):
        processes[i].join()

    print("Sample extractor done.")
