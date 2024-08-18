import pandas as pd
import numpy as np
import wfdb
from define import *
from detectors import QRSDetector_GRPC, QRSDetector_flask
from preprocessing import preprocess_data


# function for predict
def predict_flask(p_signal):
    step = 1024-145
    pred0 = np.empty((0, 2))
    pred1 = np.empty((0, 2))
    for i in range(0, len(p_signal)-step, step):
        detector = QRSDetector_flask(preprocess_data(p_signal[i:i+1024, 0]))
        detector1 = QRSDetector_flask(preprocess_data(p_signal[i:i+1024,1]))

        prediction = np.rint(detector.detect_qrs())
        prediction1 = np.rint(detector1.detect_qrs())

        pred0 = np.concatenate((pred0, prediction), axis = 0)
        pred1 = np.concatenate((pred1, prediction1), axis = 0)

    np.savetxt('tmp/pred0.txt', pred0, fmt='%d\t')
    np.savetxt('tmp/pred1.txt', pred1, fmt='%d\t')

def predict(p_signal):
    step = 1024-145
    pred0 = np.empty((0, 2))
    pred1 = np.empty((0, 2))
    for i in range(0, len(p_signal)-step, step):
        detector = QRSDetector_GRPC(preprocess_data(p_signal[i:i+1024, 0]))
        detector1 = QRSDetector_GRPC(preprocess_data(p_signal[i:i+1024,1]))

        prediction = np.rint(detector.detect_qrs())
        prediction1 = np.rint(detector1.detect_qrs())

        pred0 = np.concatenate((pred0, prediction), axis = 0)
        pred1 = np.concatenate((pred1, prediction1), axis = 0)

    np.savetxt('tmp/pred0.txt', pred0, fmt='%d\t')
    np.savetxt('tmp/pred1.txt', pred1, fmt='%d\t')
    # return np.argmax(qrs, axis=1)


# function for load data
def load_ecg(file):
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file, 'atr')
    p_signal = record.p_signal
    atr_sample = annotation.sample
    return p_signal, atr_sample

# function for load data
def load_ecg_signal(file):
    record = wfdb.rdrecord(file)
    p_signal = record.p_signal
    return p_signal


# Sample data generation for the line chart and scatter chart with different x-values
def generate_data(p_signal, atr_sample, start, end):
    # Signal
    bias = 35
    x_A = np.arange(start, start + end)/360
    value_A = p_signal[start:start + end, 0]
    value_A1 = p_signal[start:start + end, 1]

    # QRS
    x_B = [((i)/360) for i in atr_sample if start < i < start + end]
    value_B = p_signal[[i for i in atr_sample if start < i < start + end],0]
    value_B1 = p_signal[[i for i in atr_sample if start < i < start + end],1]

    # data channel 0
    signal_data = pd.DataFrame({
            'x': x_A,
            'value': value_A,
            'series': 'Signal'
        })
    qrs_data = pd.DataFrame({
        'x': x_B,
        'value': value_B,
        'series': 'QRS'
    })

    # data channel 1
    signal_data1 = pd.DataFrame({
            'x': x_A,
            'value': value_A1,
            'series': 'Signal1'
        })
    qrs_data1 = pd.DataFrame({
        'x': x_B,
        'value': value_B1,
        'series': 'QRS1'
    })

    # Predicted QRS
    try:
        qrs = np.loadtxt('tmp/pred0.txt')
        qrs1 = np.loadtxt('tmp/pred1.txt')

        sub_qrs = qrs[start:start + end]
        sub_qrs1 = qrs1[start:start + end]

        classes = np.argmax(sub_qrs, axis=1)
        classes1 = np.argmax(sub_qrs1, axis=1)

        idx_qrs = np.where(classes == 1)[0] 
        idx_qrs1 = np.where(classes1 == 1)[0]

        x_C = [(i+start+bias) / 360 for i in idx_qrs]
        value_C = value_A[idx_qrs+bias]
        x_C1 = [(i+start+bias) / 360 for i in idx_qrs1]
        value_C1 = value_A1[idx_qrs1+bias]

        pred_qrs = pd.DataFrame({
            'x': x_C,
            'value': value_C,
            'series': 'Predicted_QRS'
        })
        pred_qrs1 = pd.DataFrame({
            'x': x_C1,
            'value': value_C1,
            'series': 'Predicted_QRS1'
        })
        return pd.concat([signal_data, qrs_data, pred_qrs, signal_data1, qrs_data1, pred_qrs1])
    except:
        return pd.concat([signal_data, qrs_data, signal_data1, qrs_data1])
    
# detector = QRSDetector(preprocess_data(value_A))
# detector1 = QRSDetector(preprocess_data(value_A1))

# qrs = detector.detect_qrs()
# qrs1 = detector1.detect_qrs()

# classes = np.argmax(qrs, axis=1)
# classes1 = np.argmax(qrs1, axis=1)

# idx_qrs = np.where(classes == 1)[0] + bias
# idx_qrs1 = np.where(classes1 == 1)[0] + bias

# x_C = [(i+start) / 360 for i in idx_qrs]
# value_C = value_A[idx_qrs]
# x_C1 = [(i+start) / 360 for i in idx_qrs1]
# value_C1 = value_A1[idx_qrs1]