"""
File: grain_assembler.py

Description: Contains grain assembler tools. Workflow consists of
    1.a.) running an "assemble" function
    1.b.) performing any modifications to the assembled grains, such as interpolating a transition to another grain list
    2.a.) calculating the final grain positions using `calculate_grain_positions`
    2.b.) performing any post-calculation modifications, like changing the channel index of some grains
    3.) merging the grains to create an audio array using `merge`
"""

import aus.operations as operations
import numpy as np
import random
import grain_tools


def assemble_single(grains: list, features: list, distance_between_grains: int, max_db: float = -18.0, effect_chain: list = None, effect_cycle: list = None) -> np.ndarray:
    """
    Assembles grains given specified parameters. Applies a window norm to the final assembled grains.
    :param grains: A list of grain dictionaries to choose from
    :param feature: The string name of the audio feature to use
    :param distance_between_grains: The distance between each grain, in frames. If negative, grains will overlap. If positive, there will be a gap between grains.
    :param max_db: The db level of the final grains
    :param effect_chain: A sequence (list) of effects that will be applied to each grain individually. If None, no effects will be applied.
    :param effect_cycle: A cycle (list) of effects that will be applied to each grain in a repeating sequence. If None, no effects will be applied.
    :return: The assembled grains as an array
    """
    # Organize the grains
    for feature in features:
        grains = sorted(grains, key=lambda x: x[feature])

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
        "distance_between_grains": distance_between_grains
    }]
    for i in range(1, len(grains)):
        # Apply effects
        if effect_chain is not None:
            for effect in effect_chain:
                grains[i]["grain"] = effect(grains[i]["grain"])
        if effect_cycle is not None:
            grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        grain_pos_list.append({
            "grain": grains[i]["grain"], 
            "channel": 0,
            "distance_between_grains": distance_between_grains
            })
    
    return grain_pos_list


def assemble_stochastic(grains: list, n: int, distance_between_grains: int, rng: random.Random, max_db: float =-18.0, effect_chain: list = None, effect_cycle: list = None) -> np.ndarray:
    """
    Assembles grains stochastically. Each grain is used n times. Applies a window norm to the final assembled grains.
    :param grains: A list of grain dictionaries to choose from
    :param n: The number of occurrences of each grain
    :param rng: A random number generator object
    :param distance_between_grains: The distance between each grain, in frames. If negative, grains will overlap. If positive, there will be a gap between grains.
    :param max_db: The db level of the final grains
    :param effect_chain: A sequence (list) of effects that will be applied to each grain individually. If None, no effects will be applied.
    :param effect_cycle: A cycle (list) of effects that will be applied to each grain in a repeating sequence. If None, no effects will be applied.
    :return: The assembled grains as an array
    """
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
    grain_pos_list = [{
        "grain": grains[0]["grain"], 
        "channel": 0,
        "distance_between_grains": distance_between_grains
    }]
    for i in range(1, len(grains)):
        # Apply effects
        if effect_chain is not None:
            for effect in effect_chain:
                grains[i]["grain"] = effect(grains[i]["grain"])
        if effect_cycle is not None:
            grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        grain_pos_list.append({
            "grain": grains[i]["grain"], 
            "channel": 0,
            "distance_between_grains": distance_between_grains
            })
    
    return grain_pos_list


def assemble_repeat(grain, n: int, distance_between_grains: int, max_db: float = -18.0, effect_chain: list = None, effect_cycle: list = None) -> np.ndarray:
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
        "distance_between_grains": distance_between_grains
    }]
    for i in range(1, len(grains)):
        # Apply effects
        if effect_chain is not None:
            for effect in effect_chain:
                grains[i]["grain"] = effect(grains[i]["grain"])
        if effect_cycle is not None:
            grains[i]["grain"] = effect_cycle[i % len(effect_cycle)](grains[i]["grain"])
        grains[i]["grain"] = operations.adjust_level(grains[i]["grain"], max_db)
        grain_pos_list.append({
            "grain": grains[i]["grain"], 
            "channel": 0,
            "distance_between_grains": distance_between_grains
            })

    return grain_pos_list


def interpolate(grains1: list, grains2: list, interpolations: int) -> list:
    """
    Creates a new list of grains that interpolates linearly between two existing grain lists.
    :param grains1: A list of grains
    :param grains2: A list of grains
    :param interpolations: The number of interpolation chunk pairs
    :return: An interpolated grains list
    """
    # Calculate the slope for linear interpolation
    height1 = 2 * len(grains1) / interpolations
    height2 = 2 * len(grains2) / interpolations
    slope1 = -height1 / interpolations
    slope2 = height2 / interpolations

    # Generate the lists of alternating grains
    grains1_new = []
    grains2_new = []
    start1 = 0
    start2 = 0
    for i in range(interpolations):
        new1 = slope1 * i + height1
        new2 = slope2 * i
        end1 = min(start1 + new1, len(grains1))
        end2 = min(start2 + new2, len(grains2))
        grains1_new.append(grains1[start1:end1])
        grains2_new.append(grains2[start2:end2])
        start1 = end1
        start2 = end2
    i = 0

    # If any grains remain, pad the existing lists
    while start1 < len(grains1):
        grains1_new[i].append(grains1[start1])
        i += 1
        start1 += 1
    i = 0
    while start2 < len(grains2):
        grains2_new[i].append(grains2[start2])
        i += 1
        start2 += 1

    # Merge the grains
    newgrains = []
    for i in range(len(grains1_new)):
        newgrains += grains1_new[i]
        newgrains += grains2_new[i]
    return newgrains
    

def calculate_grain_positions(grains: list):
    """
    Calculates the actual onset position for each grain in a list of grains
    :param grains: A list of grain dictionaries
    """
    end_idx = grains[0]["grain"].shape[-1]
    grains[0]["start_idx"] = 0
    grains[0]["end_idx"] = end_idx
    for i in range(1, len(grains)):
        start_idx = end_idx + grains[i]["distance_between_grains"]
        end_idx = start_idx + grains[i]["grain"].shape[-1]
        grains[i]["start_idx"] = start_idx
        grains[i]["end_idx"] = end_idx


def merge(grains: list, num_channels: int = 1, window_fn=np.hanning) -> np.ndarray:
    """
    Merges a list of grain tuples
    :param grain_tuples: A list of tuples (grain, channel_idx, start_idx)
    :param num_channels: The number of channels
    :param window_fn: The window function
    :return: The merged array of grains
    """
    max_idx = 0
    for tup in grains:
        max_idx = max(max_idx, tup["end_idx"])
    if num_channels > 1:
        audio = np.zeros((num_channels, max_idx))
    else:
        audio = np.zeros((max_idx))
    # window_norm = np.zeros((num_channels, max_idx))
    for i in range(len(grains)):
        window = window_fn(grains[i]["grain"].shape[-1])
        grain = grains[i]["grain"] * window
        grain_tools.merge_grain(audio, grain, grains[i]["start_idx"], grains[i]["end_idx"], grains[i]["channel"])
        # grain_tools.merge(window_norm, window, tup[2], end_idx, tup[1])
    audio = np.nan_to_num(audio)
    return audio
