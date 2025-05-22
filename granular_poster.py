import mel_dev.signal_plotter as signal_plotter
import aus.audiofile as af
import aus.operations as operations
import aus.synthesis as synthesis
import scipy.fft as fft
import numpy as np

PATH = "D:\\Recording\\Samples\\granulation_chunks\\freesound_creative_commons_0\\11839__felixblume__argentina-patagonia\\131922__felixblume__argentina-tierra-del-fuego-during-the-patagonian-rodeo-the-farmers-of-the-estancia-go-into-the-fields-to-collect-the-sheep_chunk001.wav"
STARTFRAME = 144000
FFT_SIZE = 1024

# the audio file
audio_file = af.read(PATH)

# the audio chunk to work with
audio_chunk = operations.mixdown(audio_file.samples[:, STARTFRAME:STARTFRAME+FFT_SIZE])

# to plot the audio
signal_plotter.plot_signal(audio_chunk[0, :], audio_file.sample_rate, filename="data/sheep_collecting_audio.png", dpi=300, figsize=(10, 4), show_plot=False, x_axis="time", font_family="serif", font_serif="Century Schoolbook", font_size=16, color="green")

# to plot the FFT
spectrum = fft.rfft(audio_chunk[0, :])
signal_plotter.plot_spectrum(spectrum, FFT_SIZE, 44100, filename="data/sheep_collecting_spectrum1.png", dpi=300, figsize=(10, 4), show_plot=False, font_family="serif", font_serif="Century Schoolbook", font_size=16, color="red")
signal_plotter.plot_spectrum(spectrum, FFT_SIZE, 44100, frequency_range=(0, 5000),filename="data/sheep_collecting_spectrum2.png", dpi=300, figsize=(10, 4), show_plot=False, font_family="serif", font_serif="Century Schoolbook", font_size=16, color="red")

# to plot a grain
grain_srate = 10000
grain_dur = 500
grain = synthesis.sine(100, 0, grain_dur, grain_srate)
grain = np.hanning(grain_dur) * grain
signal_plotter.plot_signal(grain, grain_srate, filename="data/grain_sine_audio.png", dpi=300, figsize=(6, 3), show_plot=True, x_axis="time", font_family="serif", font_serif="Century Schoolbook", font_size=16, color="green", linewidth=2)
