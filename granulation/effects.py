"""
File: effects.py

This file contains audio effect definitions
"""

import numpy as np
import scipy.signal


class AMEffect:
    """
    A constant AM effect
    """
    def __init__(self, freqs: list, muls: list, adds: list, sample_rate: int = 44100):
        """
        Initializes the AM effect. The freqs, muls, and adds have to have the same len.
        :param freqs: A list of frequencies
        :param muls: A list of mul values (modulation depth)
        :param adds: A list of add values (shifts the modulation away from 0)
        :param sample_rate: The sample rate
        """
        self.freqs = freqs
        self.muls = muls
        self.adds = adds
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies the AM effect.
        :param audio: The audio to apply the AM effect to
        :return: The modulated audio
        """
        mod_arr = np.zeros((audio.shape[-1]))
        for i in range(len(self.freqs)):
            step = 2 * np.pi * self.freqs[i] / self.sample_rate
            x = np.arange(0, mod_arr.shape[-1] * step, step)
            mod_arr += np.sin(x) * float(self.muls[i]) + float(self.adds[i])
        if audio.ndim > 1:
            mod_arr = mod_arr.reshape((1, mod_arr.shape[-1]))
            mod_arr = mod_arr.repeat(audio.shape[0], 0)
        return audio * mod_arr


class ButterworthFilterEffect:
    """
    A Butterworth filter effect
    """
    def __init__(self, freq: float, filter_type: str = "lowpass", order: int = 1, sample_rate: int = 44100):
        """
        Initializes the ButterworthFilterEffect.
        :param freq: The cutoff frequency (if a lowpass or highpass filter), or a list of 2 frequencies (if a bandpass or bandstop filter)
        :param type: The filter type (lowpass, highpass, bandpass, bandstop)
        :param order: The filter order
        :param sample_rate: The sample rate
        """
        self.filter = scipy.signal.butter(order, freq, filter_type, False, "sos", sample_rate)
        self.filter_type = filter_type
        self.freq = freq
        self.order = order
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies the effect.
        :param audio: The audio to apply the effect to
        :return: The new audio
        """
        return scipy.signal.sosfilt(self.filter, audio)


class IdentityEffect:
    """
    A blank effect. Useful for situations where you want a placeholder in an effects list.
    """
    def __init__(self):
        pass

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        :param audio: The audio
        :return: The audio
        """
        return audio
