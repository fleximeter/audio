"""
File: dev.py

This file is for experimenting.
"""

from scipy.fft import rfft, rfftfreq, dct
from scipy.signal import ShortTimeFFT
import numpy as np
from matplotlib import pyplot as plt
# from aus import audiofile, plot
import librosa
# import signal_plotter

FFT_SIZE = 1024
AUDIO = "D:/Recording/compress.wav"
WINDOW = np.hamming(FFT_SIZE)

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

def freq(mel, htk=False):
    """
    Converts a mel to a frequency
    :param mel: The mel
    :param HTK: If true, use the HTK formula (mel = 2595 * log10(1+freq/700)). Otherwise use the Slaney piecewise formula.
    :return: The frequency
    """
    if htk:
        return 700 * (np.power(10, mel / 2595) - 1)
    else:
        # A frequency of 1000 corresponds to a mel of 15 in the Slaney formula
        if type(mel) == np.ndarray:
            freqs = np.zeros(mel.shape)
            for i in range(mel.shape[-1]):
                if mel[i] < 15:
                    freqs[i] = (200 / 3) * mel[i]
                else:
                    freqs[i] = 1000 * np.power(10, np.log10(6.4) * (mel[i] - 15) / 27)
            return freqs
        else:
            if mel < 15:
                return (200 / 3) * mel
            else:
                return 1000 * np.power(10, np.log10(6.4) * (mel - 15) / 27)

def mel(freq, htk=False):
    """
    Converts a frequency to a mel
    :param freq: The frequency
    :param HTK: If true, use the HTK formula (mel = 2595 * log10(1+freq/700)). Otherwise use the Slaney piecewise formula.
    :return: The mel
    """
    if htk:
        return 2595 * np.log10(1 + freq/700)
    
    # The Slaney formula is a bit different.
    else:
        if type(freq) == np.ndarray:
            mels = np.zeros(freq.shape)
            for i in range(freq.shape[-1]):
                if freq[i] < 1000:
                    mels[i] = 3 * freq[i] / 200
                else:
                    mels[i] = 15 + 27 * (np.log(freq[i]/1000) / np.log(6.4))
            return mels
        else:
            if freq < 1000:
                return 3 * freq / 200
            else:
                return 15 + 27 * (np.log(freq/1000) / np.log(6.4))

class Triangle:
    """
    Represents a function for computing a triangular filterbank
    """
    def __init__(self, start_pos: int, middle_pos: int, end_pos: int, low_val: float, high_val: float):
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
        

class MelFilter:
    """
    Represents a single triangular Mel filter and associated information.
    """
    def __init__(self, fft_freqs: np.ndarray, start_idx: int, end_idx: int, triangle_filter: np.ndarray, normalize=True):
        """
        Makes a new Mel filter.
        :param fft_freqs: An array of FFT frequencies
        :param start_idx: The start index in `fft_freqs` for this filter
        :param end_idx: The end index in `fft_freqs` for this filter
        :param triangle_filter: The pre-computed triangular filter
        :param normalize: Whether or not to normalize by filter width (default `True`).
        This function applies librosa-style normalization (2 / filter width)
        """
        self.fft_freqs = fft_freqs  # store a reference to the FFT frequencies array
        self.start_idx = start_idx  # we don't need to compute before the start index
        self.end_idx = end_idx      # we don't need to compute after the end index
        self.triangle_filter = triangle_filter
        if normalize:
            coef = 2 / (fft_freqs[end_idx] - fft_freqs[start_idx])
            self.triangle_filter *= coef


class MelFilterbank:
    """
    Represents a Mel filterbank
    """
    def __init__(self, low_freq, high_freq, num_filters, fft_freqs, quantize=True, normalize=True):
        """
        Creates a new Mel filterbank.
        :param filters: A list of filters
        """
        self.filterbank = MelFilterbank.make_filterbank(low_freq, high_freq, num_filters, fft_freqs, quantize, normalize)
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.fft_freqs = fft_freqs
    
    def __call__(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Applies the Mel filterbank
        :param spectrum: The spectrum to filter
        """
        filt_spec = []
        for i in range(len(self.filters)):
            filt_spec.append(np.dot(self.filters[i].triangle_filter, spectrum))
        return np.array(filt_spec)

    def make_filterbank(low_freq, high_freq, num_filters, fft_freqs, quantize=True, normalize=True):
        """
        Generates a Mel filterbank
        :param mel_start: The lowest Mel for the filterbank
        :param mel_end: The highest Mel for the filterbank
        :param num_filters: The number of filters in the filterbank
        :param fft_freqs: The FFT frequencies
        :param quantize: Whether or not to quantize the filterbank to the nearest FFT frequency
        :return: A `MelFilterbank`
        """
        # the array size is 2 larger because of endpoints 
        mel_center_freqs = np.linspace(mel(low_freq), mel(high_freq), num_filters+2)
        freq_center_freqs = np.zeros((num_filters+2))
        for i in range(num_filters+2):
            freq_center_freqs[i] = freq(mel_center_freqs[i])
        
        filterbank = []
        # make each filter
        for i in range(1, num_filters+1):
            # Make the triangle generating function
            if quantize:
                low_freq = fft_freqs[binsearch(fft_freqs, freq_center_freqs[i-1])]
                mid_freq = fft_freqs[binsearch(fft_freqs, freq_center_freqs[i])]
                high_freq = fft_freqs[binsearch(fft_freqs, freq_center_freqs[i+1])]            
                tri = Triangle(low_freq, mid_freq, high_freq, 0, 1)
            else:
                tri = Triangle(freq_center_freqs[i-1], freq_center_freqs[i], freq_center_freqs[i+1], 0, 1)
            tri_filter = np.zeros((fft_freqs.size))

            # we don't have to update each value in the array of zeros
            start_idx = binsearch_le(fft_freqs, freq_center_freqs[i-1])
            end_idx = binsearch_le(fft_freqs, freq_center_freqs[i+1])
            adjusted_end_idx = min(end_idx + 2, fft_freqs.shape[-1] - 1)
            for i in range(start_idx, adjusted_end_idx):
                tri_filter[i] = tri(fft_freqs[i])
            mel_tri_filter = MelFilter(fft_freqs, start_idx, adjusted_end_idx, tri_filter, normalize)
            filterbank.append(mel_tri_filter)
        return filterbank

def make_mel_spectrum(fb: MelFilterbank, spectrogram: np.ndarray) -> np.ndarray:
    """
    Makes a Mel spectrum
    :param fb: The filterbank
    :param spectrogram: The power spectrogram
    :return: The Mel spectrum
    """
    mspec = []
    for i in range(spectrogram.shape[-1]):
        mspec.append(fb(spectrogram[:, i]))
    mspec = np.array(mspec)
    mspec = np.swapaxes(mspec, 0, 1)
    return mspec


def plot_filterbank(fb: MelFilterbank, fft_freqs):
    """
    Plots a Mel filterbank
    :param fb: The filterbank
    :param fft_freqs: The FFT frequencies
    """
    for i in range(len(fb.filters)):
        plt.plot(fft_freqs, fb.filters[i].triangle_filter)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Mel Filterbank")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass
    # NUM_MELS = 40
    # FMIN = 64
    # FMAX = 8000
    # af = audiofile.read(AUDIO)
    # stft_ = ShortTimeFFT(np.hamming(FFT_SIZE), FFT_SIZE // 2, af.sample_rate)
    # rfreqs = rfftfreq(FFT_SIZE, 1/af.sample_rate)
    # fb = MelFilterbank(FMIN, FMAX, NUM_MELS, rfreqs, False, True)
    # ispec = stft_.stft(af.samples[0, :])
    # pspec = np.square(np.abs(ispec))
    # mel_specgram = make_mel_spectrum(fb, np.abs(ispec))
    # mel_specgram = 10 * np.log10(mel_specgram)
    # print("Custom", mel_specgram.shape)
    # signal_plotter.plot_spectrogram(mel_specgram)
    # mel_specgram2 = librosa.feature.melspectrogram(sr=af.sample_rate, S=pspec, n_fft=FFT_SIZE, hop_length=FFT_SIZE//2, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX, norm="slaney")
    # print("Librosa", mel_specgram2.shape)
    # signal_plotter.plot_spectrogram(mel_specgram2)
    # mfccs_librosa = librosa.feature.mfcc(S=mel_specgram2)
    # signal_plotter.plot_spectrogram(mfccs_librosa)

    # chunk = af.samples[0, 44100:44100+FFT_SIZE] * WINDOW
    # mag_spec = np.abs(rfft(chunk))
    # pow_spec = np.square(mag_spec)
    # filt_spec = fb(pow_spec)
    # mfccs = dct(filt_spec, 3)
    # plt.plot(mfccs)
    # plt.show()
    # plot_filterbank(fb, rfreqs)
