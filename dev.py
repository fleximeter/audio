import numpy as np
import soundfile as sf
import pedalboard as pb

with pb.io.AudioFile("/home/jeff/recording/Compositions/livepercussion/concert1.wav") as f:
    contents = f.read(f.frames)
    contents = contents * 0.5
    print(contents.shape)
    for i in range(8):
        with pb.io.AudioFile(f"/home/jeff/recording/Compositions/livepercussion/concert1_ch{i+1}.wav", 'w', samplerate=f.samplerate, num_channels=1, bit_depth=24) as outfile:
            outfile.write(contents[i, :])

