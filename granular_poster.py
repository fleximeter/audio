import signal_plotter
import aus.audiofile as af
import aus.operations as operations
import scipy.fft as fft

PATH = "D:\\Recording\\Samples\\granulation\\freesound_creative_commons_0\\260205__felixblume__toads-screaming-at-night-in-a-moat-behind-a-castle-in-belgium.wav"
STARTFRAME = 371200
FFT_SIZE = 8192

# the audio file
audio_file = af.read(PATH)

# the audio chunk to work with
audio_chunk = operations.mixdown(audio_file.samples[:, STARTFRAME:STARTFRAME+FFT_SIZE])

# to plot the audio
signal_plotter.plot_signal(audio_chunk[0, :], audio_file.sample_rate, filename="data/toads_audio.png", dpi=300, figsize=(10, 10), show_plot=False, x_axis="time", font_family="serif", font_serif="Century Schoolbook", font_size=20, color="green")

# to plot the FFT
spectrum = fft.rfft(audio_chunk[0, :])
signal_plotter.plot_spectrum(spectrum, FFT_SIZE, 44100, filename="data/toads_spectrum.png", dpi=300, figsize=(10, 10), show_plot=False, font_family="serif", font_serif="Century Schoolbook", font_size=20, color="red")
