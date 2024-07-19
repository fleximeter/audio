"""
File: grain_assembler.py

Description: Contains grain assembler tools
"""

import aus.operations as operations
import numpy as np
import random
import grain_tools


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


def repeat(grain, n: int, distance_between_grains: int, max_db: float = -18.0, effect_chain: list = None, effect_cycle: list = None) -> np.ndarray:
    """
    Repeats a grain or list of grains for n times. Applies a window norm to the final assembled grains.
    :param grain: A grain dictionary or list of grains
    :param n: The number of times to repeat
    :param distance_between_grains: The distance between each grain, in frames. If negative, grains will overlap. If positive, there will be a gap between grains.
    :param max_db: The db level of the final grains
    :param effect_chain: A sequence (list) of effects that will be applied to each grain individually. If None, no effects will be applied.
    :param effect_cycle: A cycle (list) of effects that will be applied to each grain in a repeating sequence. If None, no effects will be applied.
    :return: A list of grain tuples, specifying where each grain should go
    """
    if type(grain) == dict:
        grains = [grain] * n
    elif type(grain) == list:
        grains = grain * n

    # Apply effects
    if effect_chain is not None:
        for effect in effect_chain:
            grains[0]["grain"] = effect(grains[0]["grain"])
    if effect_cycle is not None:
        grains[0]["grain"] = effect_cycle[0](grains[0]["grain"])

    # Add the grain to the audio sequence
    grains[0]["grain"] = operations.adjust_level(grains[0]["grain"], max_db)
    grain_pos_list = [{
        "grain": grains[0]["grain"], 
        "channel": 0, 
        "start_idx": 0,
        "end_idx": grains[0]["grain"].shape[-1]
    }]
    next_start_idx = grains[0]["grain"].shape[-1]
    for i in range(1, len(grains)):
        # Apply effects
        if effect_chain is not None:
            for effect in effect_chain:
                grains[i]["grain"] = effect(grains[i]["grain"])
        if effect_cycle is not None:
            grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        start_idx = max(next_start_idx + distance_between_grains, 0)
        next_start_idx = start_idx + grains[i]["grain"].shape[-1]
        grain_pos_list.append({
            "grain": grains[i]["grain"], 
            "channel": 0, 
            "start_idx": start_idx,
            "end_idx": next_start_idx
            })

    return grain_pos_list


def merge(grain_tuples: list, num_channels: int = 1, window_fn=np.hanning) -> np.ndarray:
    """
    Merges a list of grain tuples
    :param grain_tuples: A list of tuples (grain, channel_idx, start_idx)
    :param num_channels: The number of channels
    :param window_fn: The window function
    :return: The merged array of grains
    """
    max_idx = 0
    for tup in grain_tuples:
        max_idx = max(max_idx, tup["end_idx"])
    if num_channels > 1:
        audio = np.zeros((num_channels, max_idx))
    else:
        audio = np.zeros((max_idx))
    # window_norm = np.zeros((num_channels, max_idx))
    for i in range(len(grain_tuples)):
        window = window_fn(grain_tuples[i]["grain"].shape[-1])
        grain = grain_tuples[i]["grain"] * window
        grain_tools.merge_grain(audio, grain, grain_tuples[i]["start_idx"], grain_tuples[i]["end_idx"], grain_tuples[i]["channel"])
        # grain_tools.merge(window_norm, window, tup[2], end_idx, tup[1])
    audio = np.nan_to_num(audio)
    return audio
