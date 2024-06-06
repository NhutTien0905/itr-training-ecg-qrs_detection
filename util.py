import numpy as np
from define import *
import wfdb
import os


def clustering(data):
    positive_point = np.where(data == 1)[0]
    beat = []
    if len(positive_point) > 5:
        cluster = np.array([positive_point[0]])
        for i in range(1, len(positive_point)):
            if positive_point[i] - cluster[-1] > 0.08 * FREQUENCY_SAMPLING or i == len(positive_point) - 1:
                if i == len(positive_point) - 1:
                    cluster = np.append(cluster, positive_point[i])
                if cluster.shape[0] > 5:
                    beat.append(int(np.mean(cluster)))
                cluster = np.array([positive_point[i]])
            else:
                cluster = np.append(cluster, positive_point[i])

    return np.asarray(beat)


def evaluate(file, predicted, dataset, ec57=False):
    header = wfdb.rdheader(dataset + file)
    signal_length = header.sig_len
    annotation = wfdb.rdann(dataset + file, 'atr')
    # if file.split('.')[1] == '1':
    #     annotation = wfdb.rdann(LTDB_DIR + file, 'atr', )
    # else:
    #     annotation = wfdb.rdann(LTDB_DIR + file, 'atr', )

    fs = header.fs
    if fs != FREQUENCY_SAMPLING:
        signal_length = int(FREQUENCY_SAMPLING / fs * signal_length)
        annotation.sample = annotation.sample * FREQUENCY_SAMPLING / fs
        annotation.sample = annotation.sample.astype('int')

    # condition = np.isin(annotation.symbol, ['+', '~', '|', '[', '!', ']', '"', 's', 'x'], invert=True)
    condition = np.isin(annotation.symbol,
                        ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'',
                         '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@'], invert=True)
    sample = np.extract(condition, annotation.sample)
    cluster = clustering(np.expand_dims(predicted[0:, 1], axis=1)) + int(0.1 * FREQUENCY_SAMPLING)
    window = int(0.075 * FREQUENCY_SAMPLING)

    if ec57:
        ann_dir = TEMP_DIR + dataset.split('/')[-2] + '/'
        symbol = ['N' for _ in range(sample.shape[0])]
        symbol = np.asarray(symbol)
        wfdb.wrann(file, extension='atr', sample=sample, symbol=symbol, write_dir=ann_dir, fs=250)
        symbol = ['N' for _ in range(cluster.shape[0])]
        symbol = np.asarray(symbol)
        wfdb.wrann(file, extension='pred', sample=cluster, symbol=symbol, write_dir=ann_dir, fs=250)
        return

    recording = np.zeros(signal_length, dtype='int32')
    detection = np.zeros(signal_length, dtype='int32')

    np.put(recording, sample, 1)
    np.put(detection, cluster, 1)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(sample)):
        if sum(detection[sample[i] - window:sample[i] + window + 1]) > 0:
            true_positive = true_positive + 1
        else:
            false_negative = false_negative + 1
    for i in range(len(cluster)):
        if sum(recording[cluster[i] - window:cluster[i] + window + 1]) == 0:
            false_positive = false_positive + 1

    sensitivity = true_positive / (true_positive + false_negative)
    positive_value = true_positive / (true_positive + false_positive)

    return true_positive, false_negative, false_positive, sensitivity, positive_value

