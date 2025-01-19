"""
File: mel.py

This file contains code for developing MFCC extraction.
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.fft

def freq_to_mel(freq: float) -> float:
    """
    Maps a frequency in Hz to a Mel frequency (https://en.wikipedia.org/wiki/Mel_scale)
    :param freq: The frequency to map
    :return: The Mel frequency
    """
    return 2595 * np.log10(1+freq/700)

def mel_to_freq(mel: float) -> float:
    """
    Maps a Mel frequency to a frequency in Hz (https://en.wikipedia.org/wiki/Mel_scale)
    :param mel: The Mel frequency
    :return: The frequency in Hz
    """
    return 700 * (10 ** (mel / 2595) - 1)

def compute_filterbank_indices(lower_mel: float, upper_mel: float, num_filters) -> list:
    """
    Computes Mel filterbanks (http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
    also adapted version (https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
    :param lower_mel: The lower Mel frequency
    :param upper_mel: The upper Mel frequency
    :param num_filters: The number of filterbanks
    :return: A list of filterbank tuples (lower_mel, center_mel, upper_mel)
    """
    freq_points = np.linspace(lower_mel, upper_mel, num_filters + 2)
    for i in range(freq_points.shape[-1]):
        freq_points[i] = mel_to_freq(freq_points[i])
    freq_tuples = []
    for i in range(num_filters):
        freq_tuples.append((freq_points[i], freq_points[i+1], freq_points[i+2]))
    return freq_tuples

def round_filterbank_indices(mel_filterbanks, fft_freqs) -> list:
    """
    Rounds Mel filterbanks to the closest FFT bin number
    :param mel_filterbanks: Mel filterbanks (specifying frequency, not mels)
    :param fft_freqs: A list of FFT frequencies
    :return: A rounded list of filterbank tuples, where each tuple specifies FFT bin indices
    """
    rounded_filterbanks = []
    for filterbank in mel_filterbanks:
        rounded_filterbanks.append((ordered_list_search(fft_freqs, filterbank[0]), ordered_list_search(fft_freqs, filterbank[1]), ordered_list_search(fft_freqs, filterbank[2])))
    return rounded_filterbanks

def realize_filterbanks(rounded_filterbanks, fft_size) -> list:
    """
    Realizes the filterbanks. Each filterbank is computed as a NumPy array.
    We can then implement the filter by taking the dot product of the filterbank and the FFT spectrum.
    :param rounded_filterbanks: The rounded filterbanks
    :param fft_size: The FFT size
    :return: A list of implemented filterbanks
    """
    implemented_filterbanks = []
    for fb in rounded_filterbanks:
        fb_implemented = np.zeros((fft_size // 2 + 1), dtype=np.float64)
        fb_implemented[fb[1]] = 1
        for k in range(fb[0], fb[1]):
            fb_implemented[k] = (k - fb[0]) / (fb[1] - fb[0])
        for k in range(fb[1]+1, fb[2]):
            fb_implemented[k] = (fb[2] - k) / (fb[2] - fb[1])
        implemented_filterbanks.append(fb_implemented)
    return implemented_filterbanks

def ordered_list_search(searchlist: list, target) -> int:
    """
    Searches an ordered list for the closest value. Returns the index number.
    :param searchlist: The list to search
    :param target: The target value
    :return: The index of the closest value in the list
    """
    # Edge cases
    if len(searchlist) == 0:
        return -1
    elif searchlist[0] >= target:
        return 0
    elif searchlist[-1] <= target:
        return len(searchlist) - 1
    
    # Perform log(n) search
    else:
        lower_idx = 0
        upper_idx = len(searchlist) - 1
        middle_idx = upper_idx // 2
        found = False
        while not found:
            # Base cases
            if lower_idx >= upper_idx:
                found = True
                return lower_idx
            elif searchlist[middle_idx] == target:
                found = True
                return middle_idx
            elif upper_idx - lower_idx == 1:
                if np.abs(searchlist[lower_idx] - target) < np.abs(searchlist[upper_idx] - target):
                    return lower_idx
                else:
                    return upper_idx
            
            if searchlist[middle_idx] > target:
                upper_idx = middle_idx
            else:
                lower_idx = middle_idx
            middle_idx = lower_idx + (upper_idx - lower_idx) // 2


if __name__ == "__main__":
    filter_points = []
    MEL_LOWER = freq_to_mel(20)
    MEL_UPPER = freq_to_mel(8000)
    NUM_FILTERS = 20
    FFT_SIZE = 2048
    freqs = scipy.fft.rfftfreq(FFT_SIZE, 1/44100)
    y = [0, 1, 0]
    filterbanks = compute_filterbank_indices(MEL_LOWER, MEL_UPPER, NUM_FILTERS)
    filterbanks2 = round_filterbank_indices(filterbanks, freqs)
    filterbanks3 = realize_filterbanks(filterbanks2, FFT_SIZE)
    for i in range(NUM_FILTERS):
        plt.plot(filterbanks3[i])
    plt.show()
    