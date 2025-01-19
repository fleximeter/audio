import aus.audiofile as audiofile
import numpy as np

a1 = audiofile.read("C:\\Users\\jeffr\\Downloads\\OneDrive_1_10-27-2024\\movement_1_animals.wav")
a2 = audiofile.read("C:\\Users\\jeffr\\Downloads\\OneDrive_1_10-27-2024\\movement_2_city_etc.wav")
a3 = audiofile.read("C:\\Users\\jeffr\\Downloads\\OneDrive_1_10-27-2024\\movement_3_obama.wav")

out = np.hstack((
    a1.samples,
    np.zeros((8, 44100 * 7)),
    a2.samples,
    np.zeros((8, 44100 * 7)),
    a3.samples,
    np.zeros((8, 44100 * 7))
))

a1.samples = out
audiofile.write_with_pedalboard(a1, "C:\\Users\\jeffr\\Downloads\\complete_movements.wav")\
