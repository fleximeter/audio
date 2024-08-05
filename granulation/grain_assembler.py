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


def assemble_repeat(grain, n: int, distance_between_grains: int) -> list:
    """
    Repeats a grain or list of grains for n times.
    :param grain: A grain dictionary or list of grains
    :param n: The number of times to repeat
    :param distance_between_grains: The distance between each grain, in frames. If negative, grains will overlap. If positive, there will be a gap between grains.
    :return: A list of grain tuples, specifying where each grain should go
    """
    grains = []

    # deep copy is necessary
    if type(grain) == dict:
        for _ in range(n):
            grains.append(grain.copy())
    elif type(grain) == list:
        for _ in range(n):
            for item in grain:
                grains.append(item.copy())
    
    for grain in grains:
        grain["distance_between_grains"] = distance_between_grains
        grain["channel"] = 0

    return grains


def assemble_single(grains: list, features: list, distance_between_grains: int) -> np.ndarray:
    """
    Assembles grains. Each grain is only used once. 
    Grains are sorted by features provided in the `features` list: first by feature 0, then by feature 1, etc.
    For this to work properly, you may want to round the features you are using.
    You can optionally provide an effect chain to apply to each grain, and an effect cycle where each effect is applied to the nth grain, mod the length of the effect cycle.
    :param grains: A list of grain dictionaries to choose from
    :param feature: The string name of the audio feature to use
    :param distance_between_grains: The distance between each grain, in frames. If negative, grains will overlap. If positive, there will be a gap between grains.
    :return: The assembled grains as an array
    """
    # Organize the grains
    for feature in features:
        grains = sorted(grains, key=lambda x: x[feature])

    for grain in grains:
        grain["distance_between_grains"] = distance_between_grains
        grain["channel"] = 0
    
    return grains


def assemble_stochastic(grains: list, n: int, distance_between_grains: int, rng: random.Random) -> np.ndarray:
    """
    Assembles grains stochastically. Each grain is used n times.
    :param grains: A list of grain dictionaries to choose from
    :param n: The number of occurrences of each grain
    :param rng: A random number generator object
    :param distance_between_grains: The distance between each grain, in frames. If negative, grains will overlap. If positive, there will be a gap between grains.
    :return: The assembled grains as an array
    """
    grains = grains * n
    for i in range(n):
        rng.shuffle(grains)

    for grain in grains:
        grain["distance_between_grains"] = distance_between_grains
        grain["channel"] = 0
    
    return grains


def calculate_grain_positions(grains: list):
    """
    Calculates the actual onset position for each grain in a list of grains.
    Each grain should be a dictionary with keys (grain, distance_between_grains),
    and this function will add keys (start_idx, end_idx) to each grain.
    After this function is run, you can use the `merge_grains` function to merge
    the grains into an audio array.
    :param grains: A list of grain dictionaries
    """
    end_idx = grains[0]["end_frame"] - grains[0]["start_frame"]
    grains[0]["start_idx"] = 0
    grains[0]["end_idx"] = end_idx
    for i in range(1, len(grains)):
        start_idx = end_idx + grains[i]["distance_between_grains"]
        end_idx = start_idx + grains[i]["end_frame"] - grains[i]["start_frame"]
        grains[i]["start_idx"] = start_idx
        grains[i]["end_idx"] = end_idx


def delete_nth_grains(grains: list, n: int):
    """
    Deletes every nth grain.
    :param grains: A list of grains
    :param n: Every nth grain will be deleted
    """
    i = n
    while i < len(grains):
        del grains[i]
        i += n-1


def interleave(list1: list, list2: list) -> list:
    """
    Interleaves two lists of possibly different length. The goal is to interleave as evenly as possible.
    :param list1: A list
    :param list2: A list
    :return: A combined list
    """
    # Determine which list is larger and which is smaller. Also determine the size ratio between the two lists.
    if len(list1) < len(list2):
        larger_list = list2
        smaller_list = list1
    else:
        larger_list = list1
        smaller_list = list2
    ideal_ratio = len(larger_list) / len(smaller_list)

    # The number added from each list
    list1_num_added = 0
    list2_num_added = 0

    combined_list = []

    # Add the first batch
    batch_size_1 = round(ideal_ratio)
    batch_size_2 = 1
    combined_list += list1[:batch_size_1]
    combined_list += list2[:batch_size_2]
    list1_num_added += batch_size_1
    list2_num_added += batch_size_2

    # Add until one or both lists are exhausted
    while list1_num_added < len(larger_list) and list2_num_added < len(smaller_list):
        # Fiddle with the number of items to add from each list
        temp_batch_size_1 = batch_size_1
        temp_batch_size_2 = batch_size_2
        current_ratio = list1_num_added / list2_num_added
        if current_ratio < ideal_ratio:
            temp_batch_size_1 += 1
        elif current_ratio > ideal_ratio:
            temp_batch_size_2 += 1
        temp_batch_size_1 = min(temp_batch_size_1, len(list1) - list1_num_added)
        temp_batch_size_2 = min(temp_batch_size_2, len(list2) - list2_num_added)

        # Add this batch of items
        combined_list += list1[list1_num_added:list1_num_added+temp_batch_size_1]
        combined_list += list2[list2_num_added:list2_num_added+temp_batch_size_2]
        list1_num_added += temp_batch_size_1
        list2_num_added += temp_batch_size_2
    
    # Add any remaining items
    combined_list += list1[list1_num_added:]
    combined_list += list2[list2_num_added:]

    return combined_list
    

def interpolate(grains1: list, grains2: list, interpolations=None) -> list:
    """
    Creates a new list of grains that interpolates linearly between two existing grain lists.
    :param grains1: A list of grains
    :param grains2: A list of grains
    :param interpolations: The number of interpolation chunk pairs. If None, will be determined automatically.
    This parameter can be adjusted to change the smoothness of interpolation.
    :return: An interpolated grains list
    """
    if interpolations is None:
        smaller_area = min(len(grains1), len(grains2))
        interpolations = int(np.ceil(np.sqrt(smaller_area * 2)))
    
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
        end1 = int(min(start1 + new1, len(grains1)))
        end2 = int(min(start2 + new2, len(grains2)))
        grains1_new.append(grains1[start1:end1])
        grains2_new.append(grains2[start2:end2])
        start1 = end1
        start2 = end2
    i = 0

    # If any grains remain, pad the existing lists
    while start1 < len(grains1):
        grains1_new[i].append(grains1[start1])
        i = (i + 1) % len(grains1_new)
        start1 += 1
    i = 0
    while start2 < len(grains2):
        grains2_new[i].append(grains2[start2])
        i = (i + 1) % len(grains2_new)
        start2 += 1

    # Merge the grains
    newgrains = []
    for i in range(len(grains1_new)):
        if len(grains1_new[i]) > 0 and len(grains2[i]) > 0:
            newgrains += interleave(grains1_new[i], grains2_new[i])
        else:
            newgrains += grains1_new[i] + grains2_new[i]
    
    return newgrains


def merge(grains: list, num_channels: int = 1, window_fn=np.hanning) -> np.ndarray:
    """
    Merges a list of grain dictionaries into an audio array
    :param grains: A list of grain dictionaries {grain: , start_idx: , end_idx: , channel: }
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


def merge_crossfade(grains: list, merge_fraction: float = 0.5) -> np.ndarray:
    """
    Merges several grain arrays and crossfades between them
    :param grains: A list of merged grain arrays
    :param merge_fraction: The fraction of each array that should overlap with the next array (or vice versa, depending on which array is smaller)
    :return: The merged array of grains
    """
    audio = grains[0]
    for i in range(1, len(grains)):
        audio = grain_tools.crossfade(audio, grains[i], merge_fraction)
    return audio


def randomize_param(grains: list, param: str, rng: random.Random, max_deviation: int, only_positive: bool = False):
    """
    Randomizes a grain parameter
    :param grains: A list of grain dictionaries
    :param param: The key to randomize
    :param rng: The random number generator to use
    :param max_deviation: The maximum deviation allowed
    :param only_positive: Whether or not only positive deviation is allowed
    """
    min_deviation = 0 if only_positive else -max_deviation
    for grain in grains:
        grain[param] += rng.randrange(min_deviation, max_deviation + 1)


def spread_across_channels(grains: list, num_channels: int = 2):
    """
    Spreads grains across `num_channels` channels
    :param grains: A list of grains
    :param num_channels: The number of channels
    """
    for i in range(len(grains)):
        grains[i]["channel"] = i % num_channels


def swap_nth_adjacent_pair(grains: list, n: int):
    """
    Swaps every n adjacent grain pairs.
    :param grains: A list of grains
    :param n: Every nth pair will be swapped
    """
    for i in range(0, len(grains)-1, n):
        temp = grains[i+1]
        grains[i+1] = grains[i]
        grains[i] = temp


def swap_nth_m_pair(grains: list, n: int, m: int):
    """
    Swaps every n grain pairs of grains spaced n apart
    :param grains: A list of grains
    :param n: Every nth pair will be swapped
    """
    for i in range(0, len(grains)-m, n):
        temp = grains[i+m]
        grains[i+m] = grains[i]
        grains[i] = temp


def swap_random_pair(grains: list, prob: float, rng: random.Random):
    """
    Randomly swaps adjacent grain pairs, based on the probability value provided.
    If the grains list is a list of lists, the swap will take place with the next adjacent list,
    mod the number of lists present.
    :param grains: A list of grains
    :param prob: The probability that any given pair of adjacent grains will be swapped
    :param rng: The random number generator to use
    """
    if type(grains[0]) == list:
        for i in range(len(grains)-1):
            next_idx = (i + 1) % len(grains)
            for j in range(len(grains[i])):
                swaps = rng.choices((True, False), weights=(prob, 1-prob), k=10)
                if rng.choice(swaps):
                    temp = grains[next_idx][j]
                    grains[next_idx][j] = grains[i]
                    grains[i] = temp
    else:
        for i in range(len(grains)-1):
            swaps = rng.choices((True, False), weights=(prob, 1-prob), k=10)
            if rng.choice(swaps):
                temp = grains[i]
                grains[i+1] = grains[i]
                grains[i] = temp
