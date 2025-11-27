# On-Device-LLMs
A fine-tuned LLM to import in Google's [AI Edge Gallery](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery) app on Android for offline, secure, on-device inference.

<img src="screenshot/image.png" alt="Convo with tiny gemma" width="300">

## Dataset overview
The dataset was converted from raw json to jsonl in Q&A format using a semi-automated way for an optimal input to generative models.

The original dataset can be found at: https://zenodo.org/records/15626055

## Installation
### Setup
Create a virtual environment:
````
python -m venv venv_llm
source venv_llm/bin/activate
````

Install all the packages in `requirements.txt` using the following command:
````
pip install -r requirements.txt
````

### Bundle
Due to a current dependency conflict (as of Nov 2025) between mediapipe and tensorflow/protobuf, it is advisable to decouple this bundling task from the main pipeline. 

Deactivate the previous venv and create a separate one:
````
deactivate
python -m venv venv_bundle
source venv_bundle/bin/activate
````

Then install only mediapipe in venv_bundle:
````
pip install mediapipe
````

### Hugging Face token
For gated models like gemma, you would need an [access token](https://huggingface.co/docs/hub/en/security-tokens) from Hugging Face. After creating one, store it as an environment variable.

## Running code
### Orchestration
The `pipeline.py` script is used for orchestrating the following operations:

- `train.py`: model fine-tuning.
- `val.py`: model evaluation.
- `pred.py`: model inference.
- `convert.py`: reauthor and convert the fine-tuned model from PyTorch to TFLite.
- `bundle.py`: bundle the model and tokeniser as a Task file to be imported in the AI Edge Gallery app.

### Useful commands
To train the gemma model, navigate to the root directory and use:
````
python -m src.pipeline --script train --model_name gemma
````

To bundle the model and tokeniser, change the venv as mentioned above and directly run:
````
python -m src.bundle
````

## Important notes
- Conversion to `.tflite` can take several minutes depending on the hardware, whereas bundling should take only a few seconds.
- When running the convert logic from `pipeline.py`, if there's an error related to incompatibility with jax/tensorflow/pytorch/cuda, simply comment out the import line for convert script.
- For now, only gemma model has been reauthored and quantised for on-device inference.
- The Generative API by Google is currently CPU-only, with planned support for GPU and NPU.

## Benchmarking
The following metrics were observed on my **Pixel 8a**:

| Quantisation | First Token | Latency | Prefill Speed | Decode Speed |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| INT4Block32 | 3.12 secs | 4.64 secs | 2.23 tokens/s | 28.21 tokens/s |
| INT8 | 2.13 secs | 3.72 secs | 3.29 tokens/s | 20.23 tokens/s |
