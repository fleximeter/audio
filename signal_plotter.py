"""
File: signal_plotter.py

A signal plot generator
"""

from matplotlib import pyplot as plt
import numpy as np
from aus import plot, spectrum
import scipy.fft as fft
import synthesis

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


def plot_spectrum(spectrum, sample_rate, frequency_range=None, filename=None):
    """
    Plots FFT data. The FFT data should be in original imaginary form.
    It will be converted to a normalized power spectrum in decibels.
    :param spectrum: An imaginary spectrum to plot
    :param sample_rate: The sample rate (for determining frequencies)
    :param frequency_range: If not None, only the frequencies within this range will be plotted.
    :param filename: If not None, saves to file.
    """
    fig, ax = plt.subplots()
    mags = np.abs(spectrum)
    power = np.square(mags)
    power = 20 * np.log10(np.abs(power)/np.max(np.abs(power)))
    freqs = fft.rfftfreq((spectrum.shape[-1] - 1) * 2, 1/sample_rate)
    if frequency_range is not None:
        new_freqs = []
        new_power_spectrum = []
        for i in range(freqs.shape[-1]):
            if frequency_range[0] <= freqs[i] <= frequency_range[1]:
                new_freqs.append(freqs[i])
                new_power_spectrum.append(power[i])
        ax.plot(new_freqs, new_power_spectrum)
    else:
        ax.plot(freqs, power)
    ax.set_title(f"Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    if filename is not None:
        plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # sig = synthesis.sine(100, 0, 201, 10000)
    # sig = synthesis.saw(100, 40, 10000, 10000)
    # sig = synthesis.square(100, 40, 10000, 10000)
    # sig = synthesis.triangle(100, 40, 10000, 10000)
    # sig = synthesis.am(100, [40], [1], 1000, 10000)
    sig = synthesis.ringmod(440, [40], [1], 1000, 10000)
    # sig = synthesis.fm(100, [100, 200, 300, 400, 500], [99, 199, 200, 300, 450], 1000, 10000)
    spec = fft.rfft(sig)
    plot_spectrum(spec, 10000, (0, 1000), "data/ringmod_spec.svg")
    plot_signal(sig, 10000, "time", "data/ringmod.svg")
