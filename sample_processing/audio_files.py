"""
File: audio_files.py
Author: Jeff Martin
Date: 7/16/23

This file allows easy access to a lot of audio files that I have.
"""

import audiopython.audiofile

_RECORDING_DIR = "D:\\Recording"
_IOWA_SAMPLES_DIR = "D:\\Recording\\Samples\\Iowa"

_ALTO_FLUTE_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\AltoFlute"
_BASS_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Bass.arco.mono.2444.1"
_BASS_PIZZ_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Bass.pizz.mono.2444.1"
_BASS_CLARINET_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\BassClarinet"
_BASS_FLUTE_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\BassFlute"
_BASS_TROMBONE_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\BassTrombone"
_BASSOON_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Bassoon"
_CELLO_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Cello.arco.mono.2444.1"
_CELLO_PIZZ_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Cello.pizz.mono.2444.1"
_CLARINET_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\BbClar"
_FLUTE_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\flute.nonvib"
_GUITAR_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Guitar.mono.1644.1"
_HORN_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Horn"
_OBOE_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\oboe"
_PIANO_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Piano"
_TROMBONE_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\TenorTrombone"
_TRUMPET_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Trumpet.novib"
_TUBA_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\tuba"
_VIOLA_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Viola.arco.mono.2444.1"
_VIOLA_PIZZ_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Viola.pizz.mono.2444.1"
_VIOLIN_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Violin.arco.mono.2444.1"
_VIOLIN_PIZZ_SAMPLES_DIR = f"{_IOWA_SAMPLES_DIR}\\Violin.pizz.mono.2444.1"

alto_flute_samples = audiopython.audiofile.find_files(_ALTO_FLUTE_SAMPLES_DIR)
bass_samples = audiopython.audiofile.find_files(_BASS_SAMPLES_DIR)
bass_pizz_samples = audiopython.audiofile.find_files(_BASS_PIZZ_SAMPLES_DIR)
bass_clarinet_samples = audiopython.audiofile.find_files(_BASS_CLARINET_SAMPLES_DIR)
bass_flute_samples = audiopython.audiofile.find_files(_BASS_FLUTE_SAMPLES_DIR)
bass_trombone_samples = audiopython.audiofile.find_files(_BASS_TROMBONE_SAMPLES_DIR)
bassoon_samples = audiopython.audiofile.find_files(_BASSOON_SAMPLES_DIR)
cello_samples = audiopython.audiofile.find_files(_CELLO_SAMPLES_DIR)
cello_pizz_samples = audiopython.audiofile.find_files(_CELLO_PIZZ_SAMPLES_DIR)
clarinet_samples = audiopython.audiofile.find_files(_CLARINET_SAMPLES_DIR)
flute_samples = audiopython.audiofile.find_files(_FLUTE_SAMPLES_DIR)
guitar_samples = audiopython.audiofile.find_files(_GUITAR_SAMPLES_DIR)
horn_samples = audiopython.audiofile.find_files(_HORN_SAMPLES_DIR)
oboe_samples = audiopython.audiofile.find_files(_OBOE_SAMPLES_DIR)
piano_samples = audiopython.audiofile.find_files(_PIANO_SAMPLES_DIR)
trombone_samples = audiopython.audiofile.find_files(_TROMBONE_SAMPLES_DIR)
trumpet_samples = audiopython.audiofile.find_files(_TRUMPET_SAMPLES_DIR)
tuba_samples = audiopython.audiofile.find_files(_TUBA_SAMPLES_DIR)
viola_samples = audiopython.audiofile.find_files(_VIOLA_SAMPLES_DIR)
viola_pizz_samples = audiopython.audiofile.find_files(_VIOLA_PIZZ_SAMPLES_DIR)
violin_samples = audiopython.audiofile.find_files(_VIOLIN_SAMPLES_DIR)
violin_pizz_samples = audiopython.audiofile.find_files(_VIOLIN_PIZZ_SAMPLES_DIR)
