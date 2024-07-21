"""
File: dev.py

This file is for experimenting.
"""

import aus.audiofile as audiofile
import aus.operations as operations
import aus.synthesis as synthesis
import pedalboard as pb
import datetime
import numpy as np
from granulation.grain_assembler import interleave

l1 = [i for i in range(25)]
l2 = [i for i in range(100, 110)]
l3 = interleave(l1, l2)
print(l3)
