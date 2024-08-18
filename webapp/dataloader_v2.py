import pandas as pd
import numpy as np
import wfdb
import os
from define import *
from detectors import QRSDetector
from preprocessing import preprocess_data


# function for predict
def predict(p_signal):
    step = 1024-145
    pred0 = np.empty((0, 2))
    pred1 = np.empty((0, 2))
    for i in range(0, len(p_signal)-step, step):
        detector = QRSDetector(preprocess_data(p_signal[i:i+1024, 0]))
        detector1 = QRSDetector(preprocess_data(p_signal[i:i+1024,1]))

        prediction = np.rint(detector.detect_qrs())
        prediction1 = np.rint(detector1.detect_qrs())

        pred0 = np.concatenate((pred0, prediction), axis = 0)
        pred1 = np.concatenate((pred1, prediction1), axis = 0)

    np.savetxt('tmp/pred0.txt', pred0, fmt='%d\t')
    np.savetxt('tmp/pred1.txt', pred1, fmt='%d\t')
    print('Predicted QRS saved')


# function for load data
def load_ecg_signal(file):
    record = wfdb.rdrecord(file)
    p_signal = record.p_signal
    return p_signal

def process_uploaded_files(uploaded_hea_file, uploaded_dat_file):
    # Create a temporary directory to save uploaded files
    temp_dir = "temp_ecg_data"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded files with unique names
    hea_file_path = os.path.join(temp_dir, uploaded_hea_file.name)
    dat_file_path = os.path.join(temp_dir, uploaded_dat_file.name)
    
    with open(hea_file_path, "wb") as f:
        f.write(uploaded_hea_file.getbuffer())

    with open(dat_file_path, "wb") as f:
        f.write(uploaded_dat_file.getbuffer())

    # Use the base name (without extensions) to read the record
    record_base_name = os.path.splitext(uploaded_hea_file.name)[0]
    record_path = os.path.join(temp_dir, record_base_name)
    
    # Read the record
    record = wfdb.rdrecord(record_path)
    
    # Get the signal
    signal = record.p_signal
    return signal


# Sample data generation for the line chart and scatter chart with different x-values
def generate_data(p_signal, start, end):
    # Signal
    bias = 35
    x_A = np.arange(start, start + end)/360
    value_A = p_signal[start:start + end, 0]
    value_A1 = p_signal[start:start + end, 1]

    # data channel 0
    signal_data = pd.DataFrame({
            'x': x_A,
            'value': value_A,
            'series': 'Signal'
        })

    # data channel 1
    signal_data1 = pd.DataFrame({
            'x': x_A,
            'value': value_A1,
            'series': 'Signal1'
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
        return pd.concat([signal_data, pred_qrs, signal_data1, pred_qrs1])
    except:
        return pd.concat([signal_data, signal_data1])