import tensorflow as tf
import grpc
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
# Enable GPU
physical_devices = tf.config.list_physical_devices('GPU')

class QRSDetector():
    def __init__(self, data):
        self.channel = grpc.insecure_channel("172.17.0.2:9000")
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