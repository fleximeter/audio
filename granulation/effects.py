"""
File: effects.py

This file contains audio effect definitions
"""

import numpy as np
import scipy.signal
import aus.synthesis as synthesis
import pedalboard as pb


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
            mod_arr += synthesis.sine(self.freqs[i], 0, audio.shape[-1], self.sample_rate)
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
    

class CompressorEffect:
    """
    Represents a compressor
    """
    def __init__(self, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, sample_rate: int = 44100):
        """
        Initializes the compressor
        :param threshold_db: The threshold level for the compressor
        :param ratio: The ratio of compression
        :param attack_ms: The delay before attack
        :param release_ms: The delay before release
        :param sample_rate: The sample rate
        """
        self.compressor = pb.Compressor(threshold_db, ratio, attack_ms, release_ms)
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the compressor
        :param audio: The audio
        :return: The audio
        """
        return self.compressor(audio, self.sample_rate)


class NoiseGateEffect:
    """
    Represents a noise gate
    """
    def __init__(self, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, sample_rate: int = 44100):
        """
        Initializes the noise gate
        :param threshold_db: The threshold level for the noise gate
        :param ratio: The ratio of compression
        :param attack_ms: The delay before attack
        :param release_ms: The delay before release
        :param sample_rate: The sample rate
        """
        self.noise_gate = pb.NoiseGate(threshold_db, ratio, attack_ms, release_ms)
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the noise gate
        :param audio: The audio
        :return: The audio
        """
        return self.noise_gate(audio, self.sample_rate)


class DelayEffect:
    """
    Represents a delay
    """
    def __init__(self, delay_seconds: float, feedback: float, mix: float, sample_rate: int = 44100):
        """
        Initializes the delay
        :param delay_seconds: The delay
        :param feedback: The feedback
        :param mix: The mix
        :param sample_rate: The sample rate
        """
        self.delay = pb.Delay(delay_seconds, feedback, mix)
        self.delay_seconds = delay_seconds
        self.feedback = feedback
        self.mix = mix
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the delay
        :param audio: The audio
        :return: The audio
        """
        return self.delay(audio, self.sample_rate)


class ChorusEffect:
    """
    Represents a chorus
    """
    def __init__(self, rate_hz: float, depth: float, center_delay_ms: float, feedback: float, mix: float, sample_rate: int = 44100):
        """
        Initializes the chorus
        :param rate_hz: The rate
        :param depth: The depth
        :param center_delay_ms: The center delay
        :param feedback: The feedback
        :param mix: The mix
        :param sample_rate: The sample rate
        """
        self.chorus = pb.Chorus(rate_hz, depth, center_delay_ms, feedback, mix)
        self.rate_hz = rate_hz
        self.depth = depth
        self.center_delay_ms = center_delay_ms
        self.feedback = feedback
        self.mix = mix
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the chorus
        :param audio: The audio
        :return: The audio
        """
        return self.chorus(audio, self.sample_rate)
