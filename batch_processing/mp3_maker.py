import pedalboard as pb
import os
import re
from pathlib import Path

DIR = "/home/jeff/.local/share/SuperCollider/Recordings"

pat = re.compile(r'\.(wav$)|(aiff$)', re.IGNORECASE)

for dir, subdirs, files in os.walk(DIR):
    for file in files:
        filename = Path(dir) / file
        if pat.search(file):
            with pb.io.AudioFile(str(filename), 'r') as infile:
                audio = infile.read(infile.frames)
                outpath = str(Path(filename.parent) / (filename.stem + ".mp3"))
                with pb.io.AudioFile(outpath, 'w', samplerate=infile.samplerate, num_channels=infile.num_channels, quality=160) as outfile:
                    outfile.write(audio)