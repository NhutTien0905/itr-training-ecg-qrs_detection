# Use the specified TensorFlow Serving GPU image as the base image
FROM tensorflow/serving:2.13.1-gpu

# Create necessary directories for the model and configuration files
RUN mkdir -p /tensorflow-serving/qrs_model

# Copy your model to the TensorFlow Serving container
COPY qrs_model/1 /tensorflow-serving/qrs_model/1

# Copy the configuration files to the TensorFlow Serving container
COPY model_config.txt /tensorflow-serving/model_config.txt
COPY batching_parameters.txt /tensorflow-serving/batching_parameters.txt

# Set environment variable to specify the model name
ENV MODEL_NAME=qrs

# Expose the gRPC port TensorFlow Serving uses
EXPOSE 9000

# Run TensorFlow Serving with the specified command
ENTRYPOINT ["tensorflow_model_server", \
            "--port=9000", \
            "--model_config_file=/tensorflow-serving/model_config.txt", \
            "--file_system_poll_wait_seconds=86400", \
            "--enable_batching=true", \
            "--batching_parameters_file=/tensorflow-serving/batching_parameters.txt"]
