import tensorflow as tf
import grpc
import json
import requests
from io import BytesIO
import base64
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
# Enable GPU
physical_devices = tf.config.list_physical_devices('GPU')

class QRSDetector_GRPC():
    def __init__(self, data):
        self.channel = grpc.insecure_channel("172.17.0.2:8500")
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = 'qrs'
        self.request.model_spec.signature_name = 'channels'
        self.data = data
        
    def detect_qrs(self):
        tensor_proto = tf.make_tensor_proto(self.data, dtype=np.float32, shape=self.data.shape)
        self.request.inputs['input'].CopyFrom(tensor_proto)
        try:
            result = self.stub.Predict(self.request, 30.0)
            return tf.make_ndarray(result.outputs['prediction'])
        except Exception as e:
            print(e)
            return None
        
class QRSDetector_flask():
    def __init__(self, data):
        self.data = data
        self.api_endpoint = "http://localhost:5000/qrs_detection/predict/"

    def serialize_data(self):
        # serialize the NumPy array to bytes
        buffer = BytesIO()
        np.save(buffer, self.data)  # Save the array to the buffer in .npy format
        buffer.seek(0)  # Move to the beginning of the buffer
        byte_data = buffer.read()  # Read the buffer content as bytes

        # encode the byte stream to base64
        base64_encoded_data = base64.b64encode(byte_data).decode('utf-8')
        return base64_encoded_data

    def detect_qrs(self):
        # data to be sent to api
        data = {'b64': self.serialize_data()}

        # sending post request and saving response as response object
        r = requests.post(url=self.api_endpoint, data=data)

        # convert the string to a Python object using eval
        data_list = eval(r.text)

        # convert the list to a NumPy array
        data_array = np.array(data_list)
        return data_array