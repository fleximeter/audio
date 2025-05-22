"""
File: signal_plotter.py

A signal plot generator
"""

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from aus import plot, spectrum
import scipy.fft as fft
# import synthesis

matplotlib.rcParams['font.serif'] = "Century Schoolbook"
matplotlib.rcParams['font.family'] = "serif"


def plot_signal(sig, sample_rate, **kwargs):
    """
    Plots a signal
    :param sig: The signal to plot (as a numpy array)
    :param sample_rate: The sample rate
    :param kwargs: Optional arguments
    (dpi: (a, b), filename: {None, "path"}, x_axis: {"time", "samples"})
    """
    if "color" not in kwargs:
        kwargs["color"] = "blue"
    if "dpi" not in kwargs:
        kwargs["dpi"] = 300
    if "figsize" not in kwargs:
        kwargs["figsize"] = (8, 6)
    if "filename" not in kwargs:
        kwargs["filename"] = None
    if "font_family" in kwargs:
        matplotlib.rcParams['font.family'] = kwargs["font_family"]
    if "font_sans_serif" in kwargs:
        matplotlib.rcParams['font.sans-serif'] = kwargs["font_sans_serif"]
    if "font_serif" in kwargs:
        matplotlib.rcParams['font.serif'] = kwargs["font_serif"]
    if "font_size" in kwargs:
        matplotlib.rcParams['font.size'] = kwargs["font_size"]
    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 1
    if "show_plot" not in kwargs:
        kwargs["show_plot"] = True
    if "x_axis" not in kwargs:
        kwargs["x_axis"] = "time"
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    times = np.linspace(0, sig.shape[-1] / sample_rate, sig.shape[-1])
    if kwargs["x_axis"] == "time":
        ax.plot(times, sig, linewidth=kwargs["linewidth"], color=kwargs["color"])
        ax.set_xlabel("Times (sec.)")
    else:
        ax.plot(sig, linewidth=kwargs["linewidth"], color=kwargs["color"])
        ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    if kwargs["filename"] is not None:
        plt.savefig(kwargs["filename"], dpi=kwargs["dpi"])
    if kwargs["show_plot"]:
        plt.show()


def plot_spectrum(spectrum, fft_size, sample_rate, **kwargs):
    """
    Plots FFT data. The FFT data should be in original imaginary form.
    It will be converted to a normalized power spectrum in decibels.
    :param spectrum: An imaginary spectrum to plot
    :param fft_size: The FFT size
    :param sample_rate: The sample rate (for determining frequencies)
    :param kwargs: Optional arguments
    (dpi: (a, b), filename: {None, "path"}, x_axis: {"time", "samples"})
    """
    if "color" not in kwargs:
        kwargs["color"] = "blue"
    if "dpi" not in kwargs:
        kwargs["dpi"] = 300
    if "figsize" not in kwargs:
        kwargs["figsize"] = (8, 6)
    if "filename" not in kwargs:
        kwargs["filename"] = None
    if "font_family" in kwargs:
        matplotlib.rcParams['font.family'] = kwargs["font_family"]
    if "font_sans_serif" in kwargs:
        matplotlib.rcParams['font.sans-serif'] = kwargs["font_sans_serif"]
    if "font_serif" in kwargs:
        matplotlib.rcParams['font.serif'] = kwargs["font_serif"]
    if "font_size" in kwargs:
        matplotlib.rcParams['font.size'] = kwargs["font_size"]
    if "frequency_range" not in kwargs:
        kwargs["frequency_range"] = None
    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 1
    if "show_plot" not in kwargs:
        kwargs["show_plot"] = True
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    mags = np.abs(spectrum)
    power = np.square(mags)
    power = 20 * np.log10(np.abs(power)/np.max(np.abs(power)))
    freqs = fft.rfftfreq(fft_size, 1./sample_rate)
    if kwargs["frequency_range"] is not None:
        new_freqs = []
        new_power_spectrum = []
        for i in range(freqs.shape[-1]):
            if kwargs["frequency_range"][0] <= freqs[i] <= kwargs["frequency_range"][1]:
                new_freqs.append(freqs[i])
                new_power_spectrum.append(power[i])
        ax.plot(new_freqs, new_power_spectrum, linewidth=kwargs["linewidth"], color=kwargs["color"])
    else:
        ax.plot(freqs, power, linewidth=kwargs["linewidth"], color=kwargs["color"])
    # ax.set_title(f"Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    fig.tight_layout()
    if kwargs["filename"] is not None:
        plt.savefig(kwargs["filename"], dpi=kwargs["dpi"])
    if kwargs["show_plot"]:
        plt.show()


def plot_spectrogram(spectrogram, fft_size=None, hop_size=None, sample_rate=None, mode=None, **kwargs):
    """
    Plots a STFT spectrogram. The SFFT data should be in magnitude or power form.
    :param spectrogram: A magnitude or power spectrogram to plot
    :param fft_size: The FFT size used for the spectrogram
    :param hop_size: The FFT hop size used for the spectrogram
    :param sample_rate: The sample rate (for determining frequencies)
    :param mode: The spectrogram mode (`None` just plots the raw data, `log` produces a normalized log spectrogram)
    :param kwargs: Optional arguments
    (dpi: (a, b), filename: {None, "path"}, x_axis: {"time", "samples"})
    """
    if "color" not in kwargs:
        kwargs["color"] = "blue"
    if "dpi" not in kwargs:
        kwargs["dpi"] = 300
    if "figsize" not in kwargs:
        kwargs["figsize"] = (8, 6)
    if "filename" not in kwargs:
        kwargs["filename"] = None
    if "font_family" in kwargs:
        matplotlib.rcParams['font.family'] = kwargs["font_family"]
    if "font_sans_serif" in kwargs:
        matplotlib.rcParams['font.sans-serif'] = kwargs["font_sans_serif"]
    if "font_serif" in kwargs:
        matplotlib.rcParams['font.serif'] = kwargs["font_serif"]
    if "font_size" in kwargs:
        matplotlib.rcParams['font.size'] = kwargs["font_size"]
    if "frequency_range" not in kwargs:
        kwargs["frequency_range"] = None
    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 1
    if "show_plot" not in kwargs:
        kwargs["show_plot"] = True

    fig, ax = plt.subplots(figsize=kwargs["figsize"])    
    if mode == "log":
        spectrogram = 20 * np.log10(np.abs(spectrogram)/np.max(np.abs(spectrogram)))

    ax.imshow(spectrogram, origin="lower")
    # xtick_vals = np.arange(step, step * spectrogram.shape[-1], step)
    # xtick_labels = []
    # for val in xtick_vals:
    #     if 
    # ax.set_xticks(np.arange(spectrogram.shape[-1]), np.arange(step, step * spectrogram.shape[-1], step))
    # ax.set_title(f"Spectrum")
    ax.set_xlabel("Frame no.")
    ax.set_ylabel("Frequency bin")
    fig.tight_layout()
    if kwargs["filename"] is not None:
        plt.savefig(kwargs["filename"], dpi=kwargs["dpi"])
    if kwargs["show_plot"]:
        plt.show()

if __name__ == "__main__":
    # sig = synthesis.sine(100, 0, 201, 10000)
    # sig = synthesis.saw(100, 40, 10000, 10000)
    # sig = synthesis.square(100, 40, 10000, 10000)
    # sig = synthesis.triangle(100, 40, 10000, 10000)
    # sig = synthesis.am(440, [40], [1], 1000, 10000)
    # sig = synthesis.ringmod(440, [40], [1], 1000, 10000)
    # sig = synthesis.fm(100, [100, 200, 300, 400, 500], [99, 199, 200, 300, 450], 1000, 10000)
    # spec = fft.rfft(sig)
    # plot_spectrum(spec, 10000, (0, 1000), "data/ammod_spec.svg")
    # plot_signal(sig, 10000, "time", "data/am.svg")
    pass
