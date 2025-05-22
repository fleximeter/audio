"""
Compares my frequency-to-Mel conversion with the Librosa version.
"""

import librosa
import numpy as np
import random
from mymel import mel, freq

EPSILON = 1e-8
print("Running fixed tests...")
assert(np.abs(mel(0) - librosa.core.hz_to_mel(0)) < EPSILON)
assert(np.abs(mel(3.123) - librosa.core.hz_to_mel(3.123)) < EPSILON)
assert(np.abs(mel(4.3567) - librosa.core.hz_to_mel(4.3567)) < EPSILON)
assert(np.abs(mel(5) - librosa.core.hz_to_mel(5)) < EPSILON)
assert(np.abs(mel(9.867) - librosa.core.hz_to_mel(9.867)) < EPSILON)
assert(np.abs(mel(51.877) - librosa.core.hz_to_mel(51.877)) < EPSILON)
assert(np.abs(mel(74.295) - librosa.core.hz_to_mel(74.295)) < EPSILON)
assert(np.abs(mel(100) - librosa.core.hz_to_mel(100)) < EPSILON)
assert(np.abs(mel(800) - librosa.core.hz_to_mel(800)) < EPSILON)
assert(np.abs(mel(1200) - librosa.core.hz_to_mel(1200)) < EPSILON)
assert(np.abs(mel(2000) - librosa.core.hz_to_mel(2000)) < EPSILON)
# My frequency conversion is also ok
assert(np.abs(freq(0) - librosa.core.mel_to_hz(0)) < EPSILON)
assert(np.abs(freq(3) - librosa.core.mel_to_hz(3)) < EPSILON)
assert(np.abs(freq(9) - librosa.core.mel_to_hz(9)) < EPSILON)
assert(np.abs(freq(15) - librosa.core.mel_to_hz(15)) < EPSILON)
assert(np.abs(freq(16) - librosa.core.mel_to_hz(16)) < EPSILON)
assert(np.abs(freq(21) - librosa.core.mel_to_hz(21)) < EPSILON)

# Run random tests a bunch of times
print("Running random tests...")
rng = random.Random()
rng.seed()
for _ in range(10000):
    ftom = rng.random() * 20000
    mtof = rng.random() * 30
    assert(np.abs(mel(ftom) - librosa.core.hz_to_mel(ftom)) < EPSILON)
    assert(np.abs(freq(mtof) - librosa.core.mel_to_hz(mtof)) < EPSILON)
