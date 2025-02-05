"""
File: signal_plotter.py

A signal plot generator
"""

from matplotlib import pyplot as plt
import numpy as np
from aus import synthesis

def generate_times(sample_rate, num_samples):
    """
    Generates the xlabels for time, given sample rate and number of samples
    :param sample_rate: The sample rate
    :param num_samples: The number of samples
    :return: The time array
    """
    return np.arange(0, sample_rate / num_samples, 1 / sample_rate)

def plot_signal(sig, sample_rate):
    fig, ax = plt.subplots()
    ax.plot(sig)
    ax.set_xticks([i for i in range(2000)], generate_times(sample_rate, 2000))
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    plt.show()

sig = synthesis.saw(10, 40, 2000, 10000)
plot_signal(sig)
