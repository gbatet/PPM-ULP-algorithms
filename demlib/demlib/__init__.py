# __init__.py
import numpy as np

# demlib/demlib/__init__.py

from .core import Demodulate

# Expose methods from Demodulate class for easy access
do_buffers = Demodulate.do_buffers
do_downsample_float = Demodulate.do_downsample_float
read_wav = Demodulate.read_wav
