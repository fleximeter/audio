"""
File: dev.py

This file is for experimenting.
"""

from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
from aus import audiofile

FFT_SIZE = 1024
AUDIO = "D:/Recording/compress.wav"
WINDOW = np.hamming(FFT_SIZE)
NUM_FILTERS = 10

def binsearch(arr, target) -> int:
    """
    Finds the index of the item in the array closest to the value of the target
    :param arr: The array to search
    :param target: The target to search for
    """
    if arr.size == 0:
        raise IndexError("The array is empty.")
    elif arr[0] >= target:
        return 0
    elif arr[-1] <= target:
        return arr.shape[-1] - 1
    else:
        low_idx = 0
        mid_idx = arr.size // 2
        high_idx = arr.size - 1
        while True:
            if arr[mid_idx] == target:
                return mid_idx
            elif high_idx - low_idx < 2:
                if arr[high_idx] - target >= target - arr[low_idx]:
                    return low_idx
                else:
                    return high_idx
            elif target < arr[mid_idx]:
                high_idx = mid_idx
            else:
                low_idx = mid_idx
            mid_idx = low_idx + (high_idx - low_idx) // 2

def binsearch_le(arr, target) -> int:
    """
    Finds the index of the item in the array less than or equal to the target
    :param arr: The array to search
    :param target: The target to search for
    """
    if arr.size == 0:
        raise IndexError("The array is empty.")
    elif arr[0] > target:
        raise IndexError("The target is less than the first item in the array.")
    elif arr[0] == target:
        return 0
    elif arr[-1] <= target:
        return arr.shape[-1] - 1
    else:
        low_idx = 0
        mid_idx = arr.size // 2
        high_idx = arr.size - 1
        while True:
            if arr[mid_idx] == target:
                return mid_idx
            elif high_idx - low_idx < 2:
                return low_idx
            elif target < arr[mid_idx]:
                high_idx = mid_idx
            else:
                low_idx = mid_idx
            mid_idx = low_idx + (high_idx - low_idx) // 2


def freq(mel) -> float:
    """
    Converts a mel to a frequency
    :param mel: The mel
    :return: The frequency
    """
    return 700 * (10 ** (mel/2595) - 1)

def mel(freq) -> float:
    """
    Converts a frequency to a mel
    :param freq: The frequency
    :return: The mel
    """
    return 2595 * np.log10(1 + freq/700)

class Triangle:
    """
    Represents a function for computing a triangular filterbank
    """
    def __init__(self, start_pos, middle_pos, end_pos, low_val, high_val):
        """
        Makes a Triangle function
        :param start_pos: The start X position of the triangle
        :param middle_pos: The middle X position of the triangle
        :param end_pos: The end X position of the triangle
        :param low_val: The low Y value of the triangle (end points)
        :param high_val: The high Y value of the triangle (middle point)
        """
        self.start_pos = start_pos
        self.middle_pos = middle_pos
        self.end_pos = end_pos
        self.low_val = low_val
        self.high_val = high_val
        self.ascending_slope = (high_val - low_val) / (middle_pos - start_pos)
        self.descending_slope = (low_val - high_val) / (end_pos - middle_pos)

    def __call__(self, value):
        if value <= self.start_pos or value >= self.end_pos:
            return self.low_val
        elif value <= self.middle_pos:
            return self.ascending_slope * (value - self.start_pos) + self.low_val
        else:
            return self.descending_slope * (value - self.middle_pos) + self.high_val
            

def filterbank(mel_start, mel_end, num_filters, fft_freqs, quantize=True) -> np.ndarray:
    """
    Generates a Mel filterbank
    :param mel_start: The lowest Mel for the filterbank
    :param mel_end: The highest Mel for the filterbank
    :param num_filters: The number of filters in the filterbank
    :param fft_freqs: The FFT frequencies
    :param quantize: Whether or not to quantize the filterbank to the nearest FFT frequency
    """
    # the array size is 2 larger because of endpoints 
    mel_center_freqs = np.linspace(mel_start, mel_end, num_filters+2)
    freq_center_freqs = np.zeros((num_filters+2))
    for i in range(num_filters+2):
        freq_center_freqs[i] = freq(mel_center_freqs[i])
    
    filterbank = []
    # make each filter
    for i in range(1, num_filters+1):
        if quantize:
            low_freq = fft_freqs[binsearch(fft_freqs, freq_center_freqs[i-1])]
            mid_freq = fft_freqs[binsearch(fft_freqs, freq_center_freqs[i])]
            high_freq = fft_freqs[binsearch(fft_freqs, freq_center_freqs[i+1])]            
            tri = Triangle(low_freq, mid_freq, high_freq, 0, 1)
        else:
            tri = Triangle(freq_center_freqs[i-1], freq_center_freqs[i], freq_center_freqs[i+1], 0, 1)
        filter = np.zeros((fft_freqs.size))

        # we don't have to update each value in the array of zeros
        start_idx = binsearch_le(fft_freqs, freq_center_freqs[i-1])
        end_idx = binsearch_le(fft_freqs, freq_center_freqs[i+1])
        for i in range(start_idx, min(end_idx + 2, fft_freqs.size)):
            filter[i] = tri(fft_freqs[i])
        filterbank.append(filter)
    return filterbank


if __name__ == "__main__":
    af = audiofile.read(AUDIO)
    chunk = af.samples[0, 44100:44100+FFT_SIZE] * WINDOW
    mag_spec = np.abs(rfft(chunk))
    pow_spec = np.square(mag_spec)
    rfreqs = rfftfreq(FFT_SIZE, 1/af.sample_rate)
    fb = filterbank(mel(64), mel(8000), NUM_FILTERS, rfreqs)    
    for i in range(NUM_FILTERS):
        plt.plot(fb[i])
    plt.show()
