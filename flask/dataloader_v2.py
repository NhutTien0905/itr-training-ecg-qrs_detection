import pandas as pd
import numpy as np
import wfdb
import os
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
    print('Predicted QRS saved')

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
    # record_base_name = os.path.splitext(uploaded_hea_file.name)[0]
    record_base_name = os.path.splitext(uploaded_hea_file.name)[0]
    record_path = os.path.join(temp_dir, record_base_name)
    
    # Read the record
    record = wfdb.rdrecord(record_path)
    
    # Get the signal
    signal = record.p_signal
    return signal

# preprocess index
def find_middle_indices(data, column_index, value):
    # Initialize variables
    middle_indices = []
    current_block = []
    
    # Iterate through the data to find blocks of the specified value
    for index, row in enumerate(data):
        if row[column_index] == value:
            current_block.append(index)
        else:
            if current_block:  # If the block is not empty, process it
                middle_index = len(current_block) // 2
                middle_indices.append(current_block[middle_index])
                current_block = []  # Reset for the next block

    # Process the last block if it exists
    if current_block:
        middle_index = len(current_block) // 2
        middle_indices.append(current_block[middle_index])

    return middle_indices

def average_large_gaps(middle_indices, threshold=50):
    # Initialize result list
    result_indices = []
    
    i = 0
    while i < len(middle_indices):
        # Start of a new group
        group_start = middle_indices[i]
        group_indices = [group_start]
        i += 1
        
        # Collect all indices in this group
        while i < len(middle_indices) and (middle_indices[i] - group_indices[-1] <= threshold):
            group_indices.append(middle_indices[i])
            i += 1
        
        # Calculate the average index of the group
        if group_indices:
            average_index = sum(group_indices) // len(group_indices)
            result_indices.append(average_index)
    
    return result_indices

# Sample data generation for the line chart and scatter chart with different x-values
def generate_data(p_signal, start, end):
    # Signal
    bias = 36
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

        middle_value_indices = find_middle_indices(sub_qrs, 1, 1)
        idx_qrs = np.array(average_large_gaps(middle_value_indices, threshold=50))

        middle_value_indices1 = find_middle_indices(sub_qrs1, 1, 1)
        idx_qrs1 = np.array(average_large_gaps(middle_value_indices1, threshold=50))

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