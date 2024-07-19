from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(name="grain_tools", ext_modules=cythonize(Extension("grain_tools", ["grain_tools.py"], include_dirs=[np.get_include()])))

