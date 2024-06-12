"""
File: sample_loader.py
Date: 12/28/23

A sample loader for SuperCollider. This program loads samples in a given directory
and generates a SuperCollider .scd file with information about the sample, including
loop points and code for loading a Buffer automatically.
"""

import audiopython.audiofile as audiofile
import audiopython.operations as operations
import audiopython.sampler as sampler
import sample_processing.sc_data_generator as sc_data_generator
import multiprocessing as mp
import os
import platform
import re

# Directory stuff
WINROOT = "D:\\"
MACROOT = "/Volumes/AudioJeff"
PLATFORM = platform.platform()
ROOT = WINROOT

if re.search(r'macos', PLATFORM, re.IGNORECASE):
    ROOT = MACROOT

DIR = os.path.join(ROOT, "Recording", "Samples", "Iowa", "Xylophone.hardrubber", "samples")
CPU_COUNT = mp.cpu_count()

# stuff related to this specific extraction; will need to be customized
INSTRUMENT_DICT = "xylophone.hardrubber"
DYNAMICS = {'pppp': -5, 'ppp': -4, 'pp': -3, 'p': -2, 'mp': -1, 'm': 0, 'mf': 1, 'f': 2, 'ff': 3, 'fff': 4, 'ffff': 5}
STRINGS = {'C': 0, 'G': 1, 'D': 2, 'A': 3}

sc_samples = {
    INSTRUMENT_DICT: {
        "mf": {
             "c": {},
             "g": {},
             "d": {},
             "a": {},
        },
        "ff": {
             "c": {},
             "g": {},
             "d": {},
             "a": {},
        }
    }
}


def file_processor(data_queue, files):
    """
    Loads samples from a list of files, and extracts sample information for SuperCollider.
    :param data_queue: A queue for the multiprocessing setup
    :param files: A list of files to process
    """
    local_samples = []
    for file in files:
        filename = os.path.split(file)[1]
        filename = re.sub(r'(\.wav$)|(\.aif+$)', '', filename, re.IGNORECASE)
        filename_components = filename.split('.')
        audio = audiofile.read_with_pedalboard(file)
        sample = sampler.Sample(audio.samples, 44100, file)
        sample.frames = audio.samples.shape[-1]
        sample.duration = audio.duration
        sample.pitched = True
        sample.string_name = filename_components[4][3:]
        sample.string_id = STRINGS[sample.string_name]
        sample.dynamic_name = filename_components[5]
        sample.dynamic_id = DYNAMICS[sample.dynamic_name]
        sample.instrument_name = filename_components[2].lower()
        sample.midi = filename_components[1]
        path = re.sub(r'\\', '/', sample.path)
        sample_dict = {
            "midi": sample.midi,
            "duration": sample.duration,
            "frames": sample.frames,
            "pitched": sample.pitched,
            "dynamic_name": sample.dynamic_name,
            "dynamic_id": sample.dynamic_id,
            "instrument_type": sample.instrument_name,
            "path": path,
            "string_name": sample.string_name,
            "string_id": sample.string_id,
            "buffer": f"Buffer.read(s, \"{path}\")",
            "loop_points": []
        }
        for i in range(34):
            loop_points = sampler.detect_loop_points(sample, 0, 40 - i, 0.001, 0.05, 0.1, 10000, 5000)
            if len(loop_points) > 0:
                break
        if len(loop_points) == 0:
            for i in range(14):
                loop_points = sampler.detect_loop_points(sample, 0, 20 - i, 0.001, 0.05 + i * 0.04, 0.1, 10000, 5000)
                if len(loop_points) > 0:
                    break
        # print(loop_points)
        sample_dict["loop_points"] = loop_points
        local_samples.append(sample_dict)

    # send the data back
    data_queue.put(local_samples)


if __name__ == "__main__":
    retrieved_samples = []
    files = audiofile.find_files(DIR)
    file_groups = [[] for i in range(CPU_COUNT)]
    mp_queues = [mp.Queue() for i in range(CPU_COUNT)]
    for i, file in enumerate(files):
        file_groups[i % CPU_COUNT].append(file)

    # Start the processes
    processes = []
    for i in range(CPU_COUNT):
        processes.append(mp.Process(target=file_processor, args=(mp_queues[i], file_groups[i])))
        processes[-1].start()
    
    # Collect the processes
    for i in range(CPU_COUNT):
        retrieved_samples += mp_queues[i].get()
        processes[i].join()

    # Write the sample file
    for sample in retrieved_samples:
        sc_samples[INSTRUMENT_DICT][sample["dynamic_name"]][sample["string_name"].lower()][str(sample["midi"])] = sample
        # sc_samples[INSTRUMENT_DICT][sample["dynamic_name"]][str(sample["midi"])] = sample
    with open(os.path.join(DIR, f"{INSTRUMENT_DICT}.scd"), "w") as outfile:
        outfile.write(sc_data_generator.make_sc_from_nested_objects(sc_samples))
