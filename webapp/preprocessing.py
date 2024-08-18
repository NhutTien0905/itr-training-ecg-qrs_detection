import wfdb
import numpy as np
import scipy.signal as spysig
import scipy.ndimage as spynd
from scipy.signal import lfilter, firwin
from define import *
np.seterr(divide='ignore')

# Baseline wander is a type of low-frequency noise that can affect the analysis of signals
def baseline_wander_remove(signal, fs=250, f1=0.2, f2=0.6):
    window1 = int(f1 * fs) - 1 if int(f1 * fs) % 2 == 0 else int(f1 * fs)
    window2 = int(f2 * fs) - 1 if int(f2 * fs) % 2 == 0 else int(f2 * fs)
    med_signal_0 = spysig.medfilt(signal, window1)
    med_signal_1 = spysig.medfilt(med_signal_0, window2)
    bwr_signal = signal - med_signal_1
    taps = firwin(12, 35 / fs, window='hamming')
    bwr_signal = lfilter(taps, 1.0, bwr_signal)

    return bwr_signal


def normalize(signal, fs=250, window=0.5):
    window = int(window * fs)
    max_signal_f = spynd.maximum_filter1d(signal, window)
    min_signal_f = spynd.minimum_filter1d(signal, window)
    max_signal_g = np.maximum(np.absolute(max_signal_f), np.absolute(min_signal_f))
    ave_signal = spysig.convolve(max_signal_g, np.ones(window), mode='same') / window
    bound = np.mean(ave_signal) / 2
    ave_signal = np.clip(ave_signal, a_min=bound, a_max=None)
    nor_signal = signal / ave_signal

    return nor_signal


def preprocess_data(signal):
    signal.astype(DATA_TYPE)
    signal_length = len(signal)

    signal = baseline_wander_remove(signal, FREQUENCY_SAMPLING, 0.2, 0.6)
    signal = normalize(signal, FREQUENCY_SAMPLING, 0.5)

    data_sample = []
    for i in range(NEIGHBOUR_PRE, signal_length - NEIGHBOUR_POST):
        data_sample.append(signal[i - NEIGHBOUR_PRE:i + NEIGHBOUR_POST])
    data_sample = np.expand_dims(np.array(data_sample, dtype='float32'), axis=2)
    return data_sample