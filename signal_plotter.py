"""
File: signal_plotter.py

A signal plot generator
"""

from matplotlib import pyplot as plt
import numpy as np
from aus import synthesis

def plot_signal(sig, sample_rate, x_axis="time"):
    """
    Plots a signal
    :param sig: The signal to plot (as a numpy array)
    :param sample_rate: The sample rate
    :param x_axis ("time" or "samples"): Whether the x-axis should be "time" or "samples"
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
    plt.show()

sig = synthesis.saw(10, 40, 2000, 10000)
plot_signal(sig, 10000, "samples")
