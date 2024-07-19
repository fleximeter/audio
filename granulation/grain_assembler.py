"""
File: grain_assembler.py

Description: Contains grain assembler tools
"""

import aus.operations as operations
import numpy as np
import random
from effects import *


def line(start: float, end: float, length: int) -> np.ndarray:
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


def assemble_single(grains: list, features: list, interval: int, max_db: float = -18.0) -> np.ndarray:
    """
    Assembles grains given specified parameters. Applies a window norm to the final assembled grains.
    :param grains: A list of grain dictionaries to choose from
    :param feature: The string name of the audio feature to use
    :param interval: The interval between each grain, in frames
    :param max_db: The db level of the final grains
    :return: The assembled grains as an array
    """
    effect_chain = [
        ButterworthFilterEffect(50, "highpass", 4)
    ]
    effect_cycle = [
        IdentityEffect(), 
        ChorusEffect(2, 0.5, 20, 0.4, 0.5),
        IdentityEffect(), 
        ButterworthFilterEffect(440, "lowpass", 2),
        IdentityEffect(), 
        ButterworthFilterEffect(440, "lowpass", 2),
        IdentityEffect(), 
        ChorusEffect(2, 0.5, 20, 0.4, 0.5),
    ]

    # Organize the grains
    for feature in features:
        grains = sorted(grains, key=lambda x: x[feature])

    # Create the window and initialize the window norm
    window = np.hanning(grains[0]["length"])
    window_norm = window.copy()

    # Apply effects
    for effect in effect_chain:
        grains[0]["grain"] = effect(grains[0]["grain"])
    grains[0]["grain"] = effect_cycle[0](grains[0]["grain"])

    # Add the grain to the audio sequence
    grains[0]["grain"] = operations.adjust_level(grains[0]["grain"], max_db)
    final_audio = grains[0]["grain"] * window
    last_grain_onset = 0
    for i in range(1, len(grains)):
        # Apply effects
        window = np.hanning(grains[i]["length"])
        for effect in effect_chain:
            grains[i]["grain"] = effect(grains[i]["grain"])
        grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        grains[i]["grain"] *= window

        # Add the grain to the audio sequence and update the window norm
        overlap_num = final_audio.shape[-1] - last_grain_onset - interval
        new_grain_onset = last_grain_onset + interval
        final_audio = np.hstack((final_audio[:new_grain_onset], final_audio[new_grain_onset:] + grains[i]["grain"][:overlap_num], grains[i]["grain"][overlap_num:]))
        window_norm = np.hstack((window_norm[:new_grain_onset], window_norm[new_grain_onset:] + window[:overlap_num], window[overlap_num:]))
        last_grain_onset = new_grain_onset
    
    final_audio /= window_norm
    final_audio = np.nan_to_num(final_audio)
    return final_audio


def assemble_stochastic(grains: list, n: int, interval: int, rng: random.Random, max_db: float =-18.0) -> np.ndarray:
    """
    Assembles grains stochastically. Each grain is used n times. Applies a window norm to the final assembled grains.
    :param grains: A list of grain dictionaries to choose from
    :param n: The number of occurrences of each grain
    :param rng: A random number generator object
    :param interval: The interval between each grain, in frames
    :param max_db: The db level of the final grains
    :return: The assembled grains as an array
    """
    effect_chain = [
        ButterworthFilterEffect(50, "highpass", 4)
    ]
    effect_cycle = [
        IdentityEffect(), 
        ChorusEffect(2, 0.5, 20, 0.4, 0.5),
        IdentityEffect(), 
        ButterworthFilterEffect(440, "lowpass", 2),
        IdentityEffect(), 
        ButterworthFilterEffect(440, "lowpass", 2),
        IdentityEffect(), 
        ChorusEffect(2, 0.5, 20, 0.4, 0.5),
    ]

    # Create the window and initialize the window norm
    window = np.hanning(grains[0]["length"])
    window_norm = window.copy()

    grains = grains * n
    for i in range(n):
        rng.shuffle(grains)

    # Apply effects
    for effect in effect_chain:
        grains[0]["grain"] = effect(grains[0]["grain"])
    grains[0]["grain"] = effect_cycle[0](grains[0]["grain"])

    # Add the grain to the audio sequence
    grains[0]["grain"] = operations.adjust_level(grains[0]["grain"], max_db)
    final_audio = grains[0]["grain"] * window
    last_grain_onset = 0
    for i in range(1, len(grains)):
        # Apply effects
        window = np.hanning(grains[i]["length"])
        for effect in effect_chain:
            grains[i]["grain"] = effect(grains[i]["grain"])
        grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        grains[i]["grain"] *= window

        # Add the grain to the audio sequence and update the window norm
        overlap_num = final_audio.shape[-1] - last_grain_onset - interval
        new_grain_onset = last_grain_onset + interval
        final_audio = np.hstack((final_audio[:new_grain_onset], final_audio[new_grain_onset:] + grains[i]["grain"][:overlap_num], grains[i]["grain"][overlap_num:]))
        window_norm = np.hstack((window_norm[:new_grain_onset], window_norm[new_grain_onset:] + window[:overlap_num], window[overlap_num:]))
        last_grain_onset = new_grain_onset
    
    final_audio /= window_norm
    final_audio = np.nan_to_num(final_audio)
    return final_audio
