"""
A basic phase oscillator, as seen in ECE 402
"""

import numpy as np

class PhaseOscillator:
    """
    Represents a phase oscillator
    """
    def __init__(self, memory_table_length):
        self.memory_table = np.zeros((memory_table_length))
        self.phase_register = 0
        self.freq_register = 0
        self.amp_register = 0
    
    
