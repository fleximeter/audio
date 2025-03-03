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

# sig = synthesis.sine(100, 0, 201, 10000)
# sig = synthesis.saw(100, 40, 10000, 10000)
# sig = synthesis.square(100, 40, 10000, 10000)
# sig = synthesis.triangle(100, 40, 10000, 10000)
# sig = am(100, [40, 70, 130, 160], [1, 1, 1, 1], 201, 10000)
sig = synthesis.fm(100, [100, 200, 300, 400, 500], [99, 199, 200, 300, 450], 1000, 10000)
spec = fft.rfft(sig)
plot.plot_spectrum(spec, 10000)
plot_signal(sig, 10000, "time", "data/am.svg")
