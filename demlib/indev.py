
from .utils import *
from math import ceil

#######################################################
# ID correlation
#######################################################

def do_check_pulse_broad(data, sample_rate, buffer_length, packet):

    """
    Checks if a pulse is correct within packet len in ms

    Parameters:
    data (list): Input signal detection samples.
    sample_rate (float): Sampling rate in Hz.
    buffer_length (float)
    packet (int): width time of packet in ms

    Returns:
    detec (list): List with 0-1 if the pulse is correct
    detect_times (list): Times wehere the pulse ends
    """

    input = data[0:(len(data)-1)]
    detect = []
    detect_shifted = []
    detect_times = []
    packet_len = int((packet*sample_rate)/(buffer_length*1000))

    while len(input) % packet_len:
        input.append(0)

    for i in range(0, len(data), packet_len):
        if 1 in data[(i-packet_len):i]:
            detect.append(1)
        else:
            detect.append(0)
        detect_times.append(i*buffer_length/sample_rate)

    return detect, detect_times

def correlate_id_vemco(data, packet, id_times):

    """
    Correlate sliding data with id_times

    Parameters:
    data (list): bool list of possible pings after the threshold agrupated in packet length
    packet (int in ms): size of the groping windows in ms. e.g. 340, 660...
    id_times (list): list of encoding times of the ID

    Returns:
    id_check (float) timestamp where the id is
    """

    # Make the Mask
    id_translated = [-1]*int((sum(id_times)/packet))
    id_translated[0] = 1

    for i in range(1, len(id_times)):
        index = int(sum(id_times[:i])/packet)
        id_translated[index] = 1

    id_translated.append(1)

    # correlate the mask
    check = list(np.correlate(data, id_translated, 'valid'))

    if 8 in check:
        ind = check.index(8)
        print(f"Detection at {(ind*packet/1000)} s")

    check_time = np.arange((len(id_translated)-1)*packet/1000,len(data)*packet/1000, 0.02)

    return check, check_time


def do_cfar_adapt(data, sample_rate, buffer_len, pulse_width, cells):

    """
    Implements CFAR to detect a signal

    Parameters:
    data (list): Input signal samples.
    sample_rate (float): Sampling rate in Hz.
    buffer_len (float)
    pulse_width (int): width time of the pulse to calculate the guard time
    cells (int): number of cells to calculate CFAR

    Returns:
    threshold (list): Calculated threshold for each index of the list
    detec (list): List with 0-1 if the data is under the threshold
    """

    threshold = []
    detect = []
    guard = 5*ceil(pulse_width * sample_rate / (buffer_len * 1000))
    offset = 10
    start = 0

    for i in range(len(data)):
        if data[i] == 0:
            res = 1
        elif i < (cells + guard + 1):
            res = np.mean(data[(i+guard+1):(i+guard+cells)])/data[i]

        elif i > (len(data)-cells-guard-1):
            res = np.mean(data[(i-guard-cells):(i-guard-1)])/data[i]

        else:
            do_mean= data[(i-guard-cells):(i-guard-1)] + data[(i+guard+1):(i+guard+cells)]
            res = np.mean(do_mean)/data[i]

        #Detection
        res = res-0.05
        if data[i] > res:
            detect.append(1)
        else:
            detect.append(0)

        threshold.append(res)



    return threshold, detect