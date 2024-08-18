Flask 3.0.3

- Install `tensorflow_model_server`:
```bash
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

apt-get update && apt-get install tensorflow-model-server
```
- Go to folder `flask_server`:
```bash
export FLASK_ENV=development && flask run --host=0.0.0.0
```
- Start TFServer:
```bash
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=qrs --model_base_path=/path/to/savedmodel
```
- Start app:
```bash
streamlit run app.py
```