import numpy as np
import soundfile as sf
import pedalboard as pb
import matplotlib.pyplot as plt
import scipy.signal as signal

b = [signal.butter(1, 440, 'low', output='sos', fs=44100),
signal.butter(2, 440, 'low', output='sos', fs=44100),
signal.butter(3, 440, 'low', output='sos', fs=44100),
signal.butter(4, 440, 'low', output='sos', fs=44100),
signal.butter(5, 440, 'low', output='sos', fs=44100)]
XTICKS = [22.5, 55, 110, 220, 440, 880, 1760, 3520, 7040, 14080]
XLABELS = ["A0\n(22.5)", "A1\n(55)", "A2\n(110)", "A3\n(220)", "A4\n(440)", "A5\n(880)", "A6\n(1760)", "A7\n(3520)", "A8\n(7040)", "A9\n(14080)"]

fig, ax = plt.subplots(figsize=(11, 5.5))
handles = []
labels = ["1st order", "2nd order", "3rd order", "4th order", "5th order"]
for i in range(5):
    w, h = signal.freqz_sos(b[i], 8192, fs=44100)
    handle, = ax.semilogx(w, 20 * np.log10(abs(h)), base=2, label=labels[i])
    handles.append(handle)

ax.set_xticks(XTICKS, XLABELS)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_title(f"Butterworth lowpass filters, cutoff 440 Hz")
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude (dBFS)')
ax.set_ylim(-100, 10)
ax.margins(0, 0.1)
ax.grid(which='both', axis='both')
ax.legend(handles=handles)
fig.tight_layout()
fig.savefig("/home/jeff/Documents/butter.png")