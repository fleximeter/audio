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


def fft(x: np.array, size, jump_factor, start_index) -> np.array:
    """
    The simple recursive radix 2 decimation in time FFT of a sequence x
    :param x: The sequence to perform the FFT on
    :return: The complex spectral sequence X
    """
    if size == 1:
        return np.array((x[0]), dtype=np.cdouble)
    else:
        Wn = np.exp(2j * np.pi / size)
        W = 1

        a_even = 
        # recursively compute for increasing jump factors
        X_bottom_half = fft(x, size // 2, 2 * jump_factor, 0)
        X_top_half = fft(x, size // 2, 2 * jump_factor, jump_factor)
        X = np.hstack((X_bottom_half, X_top_half), dtype=np.cdouble)

        # compute the current iteration of the FFT
        for k in range(size // 2):
            p = X[k]
            q = np.exp(-2j * np.pi * k/size) * X[k + size // 2]
            X[k] = p + q
            X[k+size // 2] = p - q
        return X
