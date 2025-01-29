import numpy as np

# NORMALIZATION

def normalize_0_1(data):
    data = data.astype(np.float32)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def normalize_neg1_1(data):
    data = data.astype(np.float32)
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1


def normalize_custom(data, a, b):
    data = data.astype(np.float32)
    return a + (data - np.min(data)) * (b - a) / (np.max(data) - np.min(data))

# DOWNSAMPLING

def do_downsample_integer(data, factor):
    """
    Downsample a signal by a specified integer factor

    Parameters:
    - data: np.array, the input signal to be downsampled
    - factor: int, the downsampling factor (every `factor`-th sample is kept)

    Returns:
    - downsampled_signal: np.array, the downsampled signal
    """
    # Check for downsampling factor
    if factor <= 1:
        raise ValueError("Downsampling factor must be greater than 1.")

    # Downsample by taking every `factor`-th sample from the filtered signal
    downsampled_signal = data[::factor]

    return downsampled_signal