# __init__.py
import numpy as np

# demlib/demlib/__init__.py

from .core import do_buffers, do_downsample_float, read_wav, do_fft, do_goertzel

from .utils import do_downsample_integer, normalize_custom, normalize_0_1, normalize_neg1_1

