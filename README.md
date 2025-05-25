# TensorFlow Learning Examples

This repository contains various examples of TensorFlow

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Examples

1. **Basic Neural Network** (`examples/01_basic_nn.py`)
   - Simple feedforward neural network for MNIST digit classification
   - Learn about layers, model building, and training

2. **CNN Image Classification** (`examples/02_cnn.py`)
   - Convolutional Neural Network for CIFAR-10 image classification
   - Learn about CNN architectures and image processing

3. **Time Series Prediction** (`examples/03_time_series.py`)
   - LSTM network for time series prediction
   - Learn about sequential data and recurrent neural networks

4. **Text Classification** (`examples/04_text_classification.py`)
   - Text classification using embeddings and LSTM
   - Learn about NLP tasks and text processing

## Running Examples

After activating the virtual environment, you can run any example:
```bash
# For macOS (using tensorflow-macos)
python3 examples/01_basic_nn.py  # Basic Neural Network
python3 examples/02_cnn.py       # CNN Image Classification
python3 examples/03_time_series.py  # Time Series Prediction
python3 examples/04_text_classification.py  # Text Classification
```

Each example contains detailed comments explaining the code and concepts. Feel free to modify the parameters and experiment with different architectures!

## Note for macOS Users
The project uses `tensorflow-macos` and `tensorflow-metal` for optimal performance on Apple Silicon chips.