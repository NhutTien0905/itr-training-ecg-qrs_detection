# importing the requests library
import argparse
import base64
import requests
import json
from preprocessing import preprocess_data
from dataloader import load_ecg_signal
import numpy as np
import requests
from io import BytesIO

# defining the api-endpoint
API_ENDPOINT = "http://localhost:5000/qrs_detection/predict/"

# preprocessing our input image
MITDB_DIR = '/home/tien/Documents/ITR/mit-bih-arrhythmia-database-1.0.0/'
p_signal = load_ecg_signal(MITDB_DIR + '100')
x_test = preprocess_data(p_signal[:,0])

# serialize the NumPy array to bytes
buffer = BytesIO()
np.save(buffer, x_test[:1,:,:])  # Save the array to the buffer in .npy format
buffer.seek(0)  # Move to the beginning of the buffer
byte_data = buffer.read()  # Read the buffer content as bytes

# encode the byte stream to base64
base64_encoded_data = base64.b64encode(byte_data).decode('utf-8')

# data to be sent to api
data = {'b64': base64_encoded_data}

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, data=data)

# convert the string to a Python object using eval
print(r.text)
# data_list = eval(r.text)

# # convert the list to a NumPy array
# data_array = np.array(data_list)
# print(np.argmax(data_array, axis=-1))