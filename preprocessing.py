import wfdb
import numpy as np
import scipy.signal as spysig
import scipy.ndimage as spynd
from scipy.signal import lfilter, firwin
from define import *
import os
import matplotlib.pyplot as plt
np.seterr(divide='ignore')


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


def preprocess_data(file_path, separate=None):
    # file_name = file_path.split('/')[-1][:-4]
    file_path = file_path[:-4]
    info = wfdb.rdheader(file_path)
    signal_length = info.sig_len
    if separate == 1:
        signal, _ = wfdb.rdsamp(file_path, channels=[0], sampfrom=0, sampto=signal_length//2)
        annotation = wfdb.rdann(file_path, 'atr', sampfrom=0, sampto=signal_length//2)
        signal_length = signal_length//2
    elif separate == 2:
        signal, _ = wfdb.rdsamp(file_path, channels=[0], sampfrom=signal_length//2, sampto=signal_length)
        annotation = wfdb.rdann(file_path, 'atr', sampfrom=signal_length//2, sampto=signal_length)
        annotation.sample = annotation.sample - (info.sig_len - info.sig_len // 2)
        signal_length = signal_length - signal_length // 2
    else:
        signal, _ = wfdb.rdsamp(file_path, channels=[0])
        annotation = wfdb.rdann(file_path, 'atr')

    signal.astype(DATA_TYPE)
    signal = np.squeeze(signal)

    if info.fs != FREQUENCY_SAMPLING:
        signal_length = int(FREQUENCY_SAMPLING / info.fs * signal_length)
        signal = spysig.resample(signal, signal_length)
        annotation.sample = annotation.sample * FREQUENCY_SAMPLING / info.fs
        annotation.sample = annotation.sample.astype('int')
        # plt.plot(np.linspace(1,100,signal[:annotation.sample[2]].shape[0]), signal[:annotation.sample[2]])
        # plt.plot(np.linspace(1,100,
        #          resample_signal[:int(annotation.sample[2] * FREQUENCY_SAMPLING / info['fs'])].shape[0]),
        #          resample_signal[:int(annotation.sample[2] * FREQUENCY_SAMPLING / info['fs'])])
        # plt.grid()
        # plt.show()
        # exit()
    signal.astype(DATA_TYPE)

    signal = baseline_wander_remove(signal, FREQUENCY_SAMPLING, 0.2, 0.6)
    signal = normalize(signal, FREQUENCY_SAMPLING, 0.5)
    # Data and labelling
    data_sample = []
    # for i in range(signal_length - (NEIGHBOUR_POINT - 1)):
    for i in range(NEIGHBOUR_PRE, signal_length - NEIGHBOUR_POST):
        data_sample.append(signal[i - NEIGHBOUR_PRE:i + NEIGHBOUR_POST])
    data_sample = np.expand_dims(np.array(data_sample, dtype='float32'), axis=2)

    label = np.zeros(signal_length, dtype='int8')
    for i in range(annotation.ann_len):
        if annotation.symbol[i] in ['+', '~', '|', '[', '!', ']', '"', 's', 'x']:
            continue
        label[annotation.sample[i] - POSITIVE_RANGE:annotation.sample[i] + POSITIVE_RANGE + 1] = 1
    label = np.array(label[int(0.1*FREQUENCY_SAMPLING):signal_length - int(0.3*FREQUENCY_SAMPLING)], dtype='int8')
    # print(np.sum(label))
    # import matplotlib.pyplot as plt
    # ranged = 1000
    # length = np.arange(len(signal[:ranged]))
    # plt.plot(length, signal[int(0.1*FREQUENCY_SAMPLING):ranged+int(0.1*FREQUENCY_SAMPLING)], label[:ranged])
    # plt.grid()
    # plt.show()
    return data_sample, label
