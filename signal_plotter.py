"""
File: signal_plotter.py

A signal plot generator
"""

from matplotlib import pyplot as plt
import numpy as np
from aus import plot, spectrum, synthesis
import scipy.fft as fft

def plot_signal(sig, sample_rate, x_axis="time", filename="file.svg"):
    """
    Plots a signal
    :param sig: The signal to plot (as a numpy array)
    :param sample_rate: The sample rate
    :param x_axis ("time" or "samples"): Whether the x-axis should be "time" or "samples"
    :param filename: The file name to save
    """
    fig, ax = plt.subplots()
    times = np.linspace(0, sig.shape[-1] / sample_rate, sig.shape[-1])
    if x_axis == "time":
        ax.plot(times, sig)
        ax.set_xlabel("Times (sec.)")
    else:
        ax.plot(sig)
        ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    plt.savefig(filename)
    plt.show()

def am(carrier_freq: int, modulator_freqs: list, modulator_amps: list, length: int, sample_rate: int) -> np.ndarray:
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

def fm(carrier_freq: int, modulator_freqs: list, modulator_depths: list, length: int, sample_rate: int) -> np.ndarray:
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

# sig = synthesis.sine(100, 0, 201, 10000)
sig = synthesis.saw(100, 40, 10000, 10000)
# sig = synthesis.square(100, 40, 10000, 10000)
# sig = synthesis.triangle(100, 40, 10000, 10000)
# sig = am(100, [40, 70, 130, 160], [1, 1, 1, 1], 201, 10000)
# sig = fm(100, [200, 300, 400, 500], [30, 10, 20, 90], 10000, 10000)
spec = fft.rfft(sig)
plot.plot_spectrum(spec, 10000)
# plot_signal(sig, 10000, "time", "data/am.svg")
