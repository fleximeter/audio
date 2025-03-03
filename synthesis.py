"""
File: synthesis.py
Contains functionality for working with different kinds of audio synthesis
"""

import numpy as np

def complex_sinusoid(freq: float, length: int, sample_rate: int) -> np.ndarray:
    """
    Makes a complex sinusoid
    :param freq: The frequency
    :param length: The length
    :param sample_rate: The sample rate
    """
    step = 2 * np.pi * freq / sample_rate
    stop = length * step
    sig = np.exp(1j * np.arange(0, stop, step))
    if sig.shape[-1] > length:
        sig = sig[:length]
    elif sig.shape[-1] < length:
        zeros = np.zeros((length - sig.shape[-1]))
        sig = np.hstack((sig, zeros))
    return sig

def freqshift(sig: np.ndarray, freq: float, sample_rate: int):
    """
    Performs frequency shifting by using a quadrature complex signal
    :param sig: The signal to shift
    :param freq: The frequency to shift by
    """

    
def am(carrier_freq: float, modulator_freqs: list, modulator_amps: list, length: int, sample_rate: int) -> np.ndarray:
    """
    Makes an AM signal
    :param carrier_freq: The carrier frequency
    :param modulator_freqs: The modulator frequencies (as a list)
    :param modulator_amps: The modulator amplitudes (as a list)
    :param len: The length of the signal
    :param sample_rate: The sample rate of the signal
    :return: The signal (as a numpy array)
    """
    # Write the signal array
    step = 2 * np.pi * carrier_freq / sample_rate
    stop = length * step
    am_sig = np.sin(np.arange(0, stop, step))
    if am_sig.shape[-1] > length:
        am_sig = am_sig[:length]
    elif am_sig.shape[-1] < length:
        zeros = np.zeros((length - am_sig.shape[-1]))
        am_sig = np.hstack((am_sig, zeros))

    # Modulate the signal
    for i in range(len(modulator_freqs)):
        step = 2 * np.pi * modulator_freqs[i] / sample_rate
        stop = length * step
        modulator = np.sin(np.arange(0, stop, step)) * modulator_amps[i]
        if am_sig.shape[-1] > length:
            am_sig = am_sig[:length]
        elif am_sig.shape[-1] < length:
            zeros = np.zeros((length - am_sig.shape[-1]))
            am_sig = np.hstack((am_sig, zeros))
        am_sig *= modulator
    
    return am_sig

def fm(carrier_freq: float, modulator_freqs: list, modulator_depths: list, length: int, sample_rate: int) -> np.ndarray:
    """
    Makes a FM signal
    :param carrier_freq: The carrier frequency
    :param modulator_freqs: The modulator frequencies (as a list)
    :param modulator_depths: The modulator depths (as a list)
    :param len: The length of the signal
    :param sample_rate: The sample rate of the signal
    :return: The signal (as a numpy array)
    """
    # Start by writing the modulator array
    modulator = np.zeros((length), dtype=np.float64) + carrier_freq
    for i in range(len(modulator_freqs)):
        step = 2 * np.pi * modulator_freqs[i] / sample_rate
        stop = length * step
        fm_sig = np.sin(np.arange(0, stop, step))
        if fm_sig.shape[-1] > length:
            fm_sig = fm_sig[:length]
        elif fm_sig.shape[-1] < length:
            zeros = np.zeros((length - fm_sig.shape[-1]))
            fm_sig = np.hstack((fm_sig, zeros))
        modulator += fm_sig * modulator_depths[i]
    
    # Write the signal array
    phases = np.zeros((length), dtype=np.float64)
    cphase = 0
    for i in range(length):
        phases[i] = cphase
        cphase += 2 * np.pi * modulator[i] / sample_rate
    fm_sig = np.sin(phases)
    if fm_sig.shape[-1] > length:
        fm_sig = fm_sig[:length]
    elif fm_sig.shape[-1] < length:
        zeros = np.zeros((length - fm_sig.shape[-1]))
        fm_sig = np.hstack((fm_sig, zeros))

    return fm_sig


def saw(freq: float, max_harmonic: int, length: int, sample_rate: int = 44100):
    """
    Generates a sawtooth tone
    :param freq: The frequency
    :param max_harmonic: The maximum harmonic index
    :param len: The length of the signal
    :param sample_rate: The audio sample rate
    :return: The sawtooth signal
    """
    sig = np.zeros((length))
    for harmonic in range(1, max_harmonic + 1):
        step = 2 * np.pi * freq * harmonic / sample_rate
        stop = length * step
        harmonic_sig = np.sin(np.arange(0, stop, step))
        if harmonic_sig.shape[-1] > length:
            harmonic_sig = harmonic_sig[:length]
        elif harmonic_sig.shape[-1] < length:
            zeros = np.zeros((length - harmonic_sig.shape[-1]))
            harmonic_sig = np.hstack((harmonic_sig, zeros))
        sig += 1 / (2 * harmonic) * harmonic_sig
    return sig


def sine(freq: float, phase: float, length: int, sample_rate: int = 44100):
    """
    Generates a sine tone
    :param freq: The frequency
    :param phase: The phase
    :param len: The length of the signal
    :param sample_rate: The audio sample rate
    :return: The sine signal
    """
    step = 2 * np.pi * freq / sample_rate
    stop = length * step + phase
    sig = np.sin(np.arange(phase, stop, step))
    if sig.shape[-1] > length:
        sig = sig[:length]
    elif sig.shape[-1] < length:
        zeros = np.zeros((length - sig.shape[-1]))
        sig = np.hstack((sig, zeros))
    return sig


def square(freq: float, max_harmonic: int, length: int, sample_rate: int = 44100):
    """
    Generates a square tone
    :param freq: The frequency
    :param max_harmonic: The maximum harmonic index
    :param len: The length of the signal
    :param sample_rate: The audio sample rate
    :return: The square signal
    """
    max_harmonic = (max_harmonic - 1) // 2
    sig = np.zeros((length))
    for harmonic in range(0, max_harmonic + 1):
        step = 2 * np.pi * freq * (2 * harmonic + 1) / sample_rate
        stop = length * step
        harmonic_sig = np.sin(np.arange(0, stop, step))
        if harmonic_sig.shape[-1] > length:
            harmonic_sig = harmonic_sig[:length]
        elif harmonic_sig.shape[-1] < length:
            zeros = np.zeros((length - harmonic_sig.shape[-1]))
            harmonic_sig = np.hstack((harmonic_sig, zeros))
        sig += 1 / (2 * harmonic + 1) * harmonic_sig
    return sig


def triangle(freq: float, max_harmonic: int, length: int, sample_rate: int = 44100):
    """
    Generates a triangle tone
    :param freq: The frequency
    :param max_harmonic: The maximum harmonic index
    :param len: The length of the signal
    :param sample_rate: The audio sample rate
    :return: The triangle signal
    """
    max_harmonic = (max_harmonic - 1) // 2
    sig = np.zeros((length))
    for harmonic in range(0, max_harmonic + 1):
        step = 2 * np.pi * freq * (2 * harmonic + 1) / sample_rate
        stop = length * step
        harmonic_sig = np.sin(np.arange(0, stop, step))
        if harmonic_sig.shape[-1] > length:
            harmonic_sig = harmonic_sig[:length]
        elif harmonic_sig.shape[-1] < length:
            zeros = np.zeros((length - harmonic_sig.shape[-1]))
            harmonic_sig = np.hstack((harmonic_sig, zeros))
        sig += (-1) ** harmonic / (2 * harmonic + 1) ** 2 * harmonic_sig
    sig = sig * 8 / np.pi ** 2
    return sig
