"""
File: fft.py

DFT and FFT implementations for fun
"""

import numpy as np


def dft(x: np.array) -> np.array:
    """
    The simple DFT of a sequence x
    :param x: The sequence to perform the DFT on
    :return: The complex spectral sequence X
    """
    spectrum = np.zeros(x.shape, dtype=np.cdouble)
    for k in range(x.size):
        const = -2j * np.pi * k / x.size
        for n in range(x.size):
            spectrum[k] += x[n] * np.exp(const * n)
    return spectrum


def idft(X: np.array) -> np.array:
    """
    The IDFT of a spectral sequence X
    :param X: The complex spectral sequence X
    :return: The sequence x
    """
    spectrum = np.zeros(X.shape, dtype=np.int64)
    for n in range(X.size):
        const = -2j * np.pi * n / X.size
        for k in range(X.size):
            spectrum[n] += X[k] * np.exp(const * k)
        spectrum[n] /= X.size
    return spectrum
