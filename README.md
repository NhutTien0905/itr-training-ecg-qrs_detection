# I. Setup environment
### 1. Install libraries
- Install tensorflow 2.16.1
```bash
pip install tensorflow==2.16.1
pip install tensorflow-serving-api
```
- Install grpc
```bash
pip install grpcio
```
### 2. Install CUDA 12.3 and CUDNN 8.9
- To verify your gpu is cuda enable check
```bash
lspci | grep -i nvidia
```
- If you have previous installation remove it first. 
```bash
sudo apt purge nvidia* -y
sudo apt remove nvidia-* -y
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt autoremove -y && sudo apt autoclean -y
sudo rm -rf /usr/local/cuda*
```
- System update
```bash
sudo apt update && sudo apt upgrade -y
```
- Install other import packages
```bash
sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```
- First get the PPA repository driver
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```
- Find recommended driver versions for you
```bash
ubuntu-drivers devices
```
- Install nvidia driver with dependencies
```bash
sudo apt install libnvidia-common-555 libnvidia-gl-555 nvidia-driver-555 -y
```
- Reboot
```bash
sudo reboot now
```
- Verify that the following bash works
```bash
nvidia-smi
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
```
- Update and upgrade
```bash
sudo apt update && sudo apt upgrade -y
```
- Installing CUDA-12.3
```bash
sudo apt install cuda-12-3 -y
```
- Setup your paths
```bash
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```
- Install cuDNN 8.9 - First register here: https://developer.nvidia.com/developer-program/signup
```bash
CUDNN_TAR_FILE="cudnn-linux-x86_64-8.9.0.131_cuda12-archive.tar.xz"
sudo tar -xvf ${CUDNN_TAR_FILE}
sudo mv cudnn-linux-x86_64-8.9.0.131_cuda12-archive cuda
```
- Copy the following files into the cuda toolkit directory.
```bash
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-12.3/include
sudo cp -P cuda/lib/libcudnn* /usr/local/cuda-12.3/lib64/
sudo chmod a+r /usr/local/cuda-12.3/lib64/libcudnn*
```
- Finally, to verify the installation, check
```bash
nvidia-smi
nvcc -V
```
### 3. Install NVIDIA Container Toolkit
- Update and install
```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit-base
```
- Check version
```bash
nvidia-ctk --version
```
- Generate a CDI specification
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```
- Check name of generated devices
```bash
grep "  name:" /etc/cdi/nvidia.yaml
```
### 4. Install Docker
- Install lastest version
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
- Check installation
```bash
sudo docker run hello-world
```

# II. Convert model and Compare with .h5 version
### 1. Convert model
```python
import os
import tensorflow as tf
# disable using GPU
tf.config.set_visible_devices([], 'GPU')
# input shape
SHAPE = (145, 1)
# configuration
TF_CONFIG = {
    'model_name': 'qrs',
    'signature': 'channels',
    'input': 'input',
    'output': 'prediction',
}

class ExportModel(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, *SHAPE), dtype=tf.float32),])
    def score(self, input: tf.TensorSpec) -> dict:
        result = self.model([{
            TF_CONFIG['input']: input,
        }])
        return {
            TF_CONFIG['output']: result
        }
    
def export_model(model, output_path):
    os.makedirs(output_path, exist_ok=True)
    module = ExportModel(model)
    batched_module = tf.function(module.score)
    tf.saved_model.save(
        module,
        output_path,
        signatures={
            TF_CONFIG['signature']: batched_module.get_concrete_function(
                tf.TensorSpec(shape=(None, *SHAPE), dtype=tf.float32),
            )
        }
    )

def main(model_dir):
    model = tf.keras.models.load_model("path/to/*.h5")
    model_dir = f'{model_dir}/version'
    os.makedirs(model_dir, exist_ok=True)
    export_model(model=model, output_path=model_dir)

if __name__ == '__main__':
    model_dir = ''
    main("save/model/path")
```
### 2. Compare result

# III. Import model
### 1. Load docker image
- Pull docker
```bash
docker pull tensorflow/serving:2.16.1
```
- Run image (init container)
```bash
sudo docker run --name=tf-gpu2 -it --entrypoint=/bin/bash tensorflow/serving:2.16.1
sudo docker start -i tf-gpu2
```
- Create folder
```bash
mkdir tensorflow-serving
cd tensorflow-serving
mkdir qrs_model
```
- Create dependence files
```bash
cd ..
touch model_config.txt
cat > model_config.txt << EOL
model_config_list: {
  config: {
    name: "qrs",
    base_path: "/tensorflow-serving/qrs_model",
    model_platform: "tensorflow"
  }
}
EOL
touch batching_parameters.txt
cat > batching_parameters << EOL
max_batch_size { value: 1024 }
batch_timeout_micros { value: 1000 }
num_batch_threads { value: 12 }
pad_variable_length_inputs: true
EOL
```
- Copy to TFServer
```bash
sudo docker cp /home/tien/Documents/ITR/itr-training-ecg-qrs_detection/model_new/1 tf-gpu2:/tensorflow-serving/qrs_model
```
- Start server
```bash
tensorflow_model_server --port=9000 --model_config_file=/tensorflow-serving/model_config.txt
```
### 2. Use server
- Check configuration of converted model
```bash
saved_model_cli show --dir /home/tien/Documents/ITR/itr-training-ecg-qrs_detection/model_new/1 --all
```
- Python code
```python
from preprocessing import preprocess_data
import tensorflow as tf
import grpc
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# Ensure compatibility with TensorFlow 1.x API
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set visible devices to CPU only
tf.config.set_visible_devices([], 'GPU')

# Establish gRPC channel
channel = grpc.insecure_channel("172.17.0.2:9000")
grpc.channel_ready_future(channel).result(timeout=10)
print("Connected to gRPC server")

# Create gRPC stub
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create PredictRequest
request = predict_pb2.PredictRequest()
request.model_spec.name = "qrs"
request.model_spec.signature_name = "channels"

def grpc_infer(imgs):
    print(imgs.shape)
    tensor_proto = tf.make_tensor_proto(imgs, dtype=tf.float32, shape=imgs.shape)
    request.inputs["input"].CopyFrom(tensor_proto)

    try:
        result = stub.Predict(request, 30.0)
        result = result.outputs["prediction"]
        return result
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Preprocess data
MITDB_DIR = '/home/tien/Documents/ITR/mit-bih-arrhythmia-database-1.0.0/'
test_data, _ = preprocess_data(MITDB_DIR + '100.hea')

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    y_pred = grpc_infer(test_data)

print(y_pred)
```
