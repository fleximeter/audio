"""
File: grain_assembler.py

Description: Contains grain assembler tools
"""

import aus.operations as operations
import numpy as np
import random


def assemble_single(grains: list, features: list, interval: int, max_db: float = -18.0, window_fn = np.hanning, effect_chain: list = None, effect_cycle: list = None) -> np.ndarray:
    """
    Assembles grains given specified parameters. Applies a window norm to the final assembled grains.
    :param grains: A list of grain dictionaries to choose from
    :param feature: The string name of the audio feature to use
    :param interval: The interval between each grain, in frames
    :param max_db: The db level of the final grains
    :param window_fn: A window function to apply to all grains
    :param effect_chain: A sequence (list) of effects that will be applied to each grain individually. If None, no effects will be applied.
    :param effect_cycle: A cycle (list) of effects that will be applied to each grain in a repeating sequence. If None, no effects will be applied.
    :return: The assembled grains as an array
    """
    # Organize the grains
    for feature in features:
        grains = sorted(grains, key=lambda x: x[feature])

    # Create the window and initialize the window norm
    window = window_fn(grains[0]["length"])
    window_norm = window.copy()

    # Apply effects
    if effect_chain is not None:
        for effect in effect_chain:
            grains[0]["grain"] = effect(grains[0]["grain"])
    if effect_cycle is not None:
        grains[0]["grain"] = effect_cycle[0](grains[0]["grain"])

    # Add the grain to the audio sequence
    grains[0]["grain"] = operations.adjust_level(grains[0]["grain"], max_db)
    final_audio = grains[0]["grain"] * window
    last_grain_onset = 0
    for i in range(1, len(grains)):
        # Apply effects
        if effect_chain is not None:
            for effect in effect_chain:
                grains[i]["grain"] = effect(grains[i]["grain"])
        if effect_cycle is not None:
            grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        window = window_fn(grains[i]["length"])
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


def assemble_stochastic(grains: list, n: int, interval: int, rng: random.Random, max_db: float =-18.0, window_fn = np.hanning, effect_chain: list = None, effect_cycle: list = None) -> np.ndarray:
    """
    Assembles grains stochastically. Each grain is used n times. Applies a window norm to the final assembled grains.
    :param grains: A list of grain dictionaries to choose from
    :param n: The number of occurrences of each grain
    :param rng: A random number generator object
    :param interval: The interval between each grain, in frames
    :param max_db: The db level of the final grains
    :param window_fn: A window function to apply to all grains
    :param effect_chain: A sequence (list) of effects that will be applied to each grain individually. If None, no effects will be applied.
    :param effect_cycle: A cycle (list) of effects that will be applied to each grain in a repeating sequence. If None, no effects will be applied.
    :return: The assembled grains as an array
    """
    # Create the window and initialize the window norm
    window = window_fn(grains[0]["length"])
    window_norm = window.copy()

    grains = grains * n
    for i in range(n):
        rng.shuffle(grains)

    # Apply effects
    if effect_chain is not None:
        for effect in effect_chain:
            grains[0]["grain"] = effect(grains[0]["grain"])
    if effect_cycle is not None:
        grains[0]["grain"] = effect_cycle[0](grains[0]["grain"])

    # Add the grain to the audio sequence
    grains[0]["grain"] = operations.adjust_level(grains[0]["grain"], max_db)
    final_audio = grains[0]["grain"] * window
    last_grain_onset = 0
    for i in range(1, len(grains)):
        # Apply effects
        if effect_chain is not None:
            for effect in effect_chain:
                grains[i]["grain"] = effect(grains[i]["grain"])
        if effect_cycle is not None:
            grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        window = window_fn(grains[i]["length"])
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


def repeat(grain, n: int, distance_between_grains: int, max_db: float = -18.0, window_fn = np.hanning, effect_chain: list = None, effect_cycle: list = None) -> np.ndarray:
    """
    Repeats a grain or list of grains for n times. Applies a window norm to the final assembled grains.
    :param grain: A grain dictionary or list of grains
    :param n: The number of times to repeat
    :param distance_between_grains: The distance between each grain, in frames. If negative, grains will overlap. If positive, there will be a gap between grains.
    :param max_db: The db level of the final grains
    :param window_fn: A window function to apply to all grains
    :param effect_chain: A sequence (list) of effects that will be applied to each grain individually. If None, no effects will be applied.
    :param effect_cycle: A cycle (list) of effects that will be applied to each grain in a repeating sequence. If None, no effects will be applied.
    :return: The assembled grains as an array
    """
    if type(grain) == dict:
        grains = [grain] * n
    elif type(grain) == list:
        grains = grain * n

    # Create the window and initialize the window norm
    window = window_fn(grains[0]["length"])
    window_norm = window.copy()

    # Apply effects
    if effect_chain is not None:
        for effect in effect_chain:
            grains[0]["grain"] = effect(grains[0]["grain"])
    if effect_cycle is not None:
        grains[0]["grain"] = effect_cycle[0](grains[0]["grain"])

    # Add the grain to the audio sequence
    grains[0]["grain"] = operations.adjust_level(grains[0]["grain"], max_db)
    final_audio = grains[0]["grain"] * window
    for i in range(1, len(grains)):
        # Apply effects
        if effect_chain is not None:
            for effect in effect_chain:
                grains[i]["grain"] = effect(grains[i]["grain"])
        if effect_cycle is not None:
            grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        window = window_fn(grains[i]["length"])
        grains[i]["grain"] *= window

        # Add the grain to the audio sequence and update the window norm
        if distance_between_grains > 0:
            zeros_shape = list(grains[i]["grain"].shape)
            zeros_shape[-1] = distance_between_grains
            final_audio = np.hstack((final_audio, np.zeros(zeros_shape), grains[i]["grain"]))
        else:
            final_audio = np.hstack((final_audio[:final_audio.shape[-1] + distance_between_grains], 
                                     final_audio[final_audio.shape[-1] + distance_between_grains:] + grains[i]["grain"][:distance_between_grains * -1], 
                                     grains[i]["grain"][distance_between_grains * -1:]))
            window_norm = np.hstack((window_norm[:window_norm.shape[-1] + distance_between_grains], 
                                     window_norm[window_norm.shape[-1] + distance_between_grains:] + window[:distance_between_grains * -1], 
                                     window[distance_between_grains * -1:]))
    
    if distance_between_grains < 0:
        final_audio /= window_norm
    final_audio = np.nan_to_num(final_audio)
    return final_audio
