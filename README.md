# Quantization-Aware Training (QAT) Project

A modern implementation of Quantization-Aware Training using PyTorch 2.0+ APIs, featuring comprehensive logging, visualization, and a web-based dashboard.

## Features

- **Modern PyTorch 2.0+ APIs**: Uses `torch.ao.quantization` instead of deprecated APIs
- **Advanced Architecture**: MLP with BatchNorm, Dropout, and dynamic layer construction
- **Comprehensive Logging**: TensorBoard integration, SQLite database for experiment tracking
- **Web Dashboard**: Streamlit-based UI for training management and visualization
- **Mock Database**: SQLite-based experiment and metrics storage
- **Configurable Training**: YAML-based configuration management
- **Real-time Monitoring**: Live training progress and metrics visualization
- **Model Analysis**: Comprehensive analytics and performance insights

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for complete dependencies

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Quantization-Aware-Training.git
cd Quantization-Aware-Training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Training

1. **Basic training**:
```bash
python qat_trainer.py
```

2. **Training with visualizations**:
```bash
python qat_trainer.py --visualize
```

3. **Custom configuration**:
```bash
python qat_trainer.py --config custom_config.yaml --visualize
```

### Web Dashboard

1. **Launch the Streamlit app**:
```bash
streamlit run app.py
```

2. **Open your browser** to `http://localhost:8501`

## Project Structure

```
├── qat_trainer.py          # Main training script
├── app.py                  # Streamlit web dashboard
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── test_qat.py            # Test suite
├── 0155.py               # Original implementation
├── data/                 # MNIST dataset (auto-downloaded)
├── logs/                 # Training logs and TensorBoard files
├── models/               # Saved model checkpoints
├── plots/                # Generated visualizations
└── qat_results.db        # SQLite database for experiments
```

## Configuration

The `config.yaml` file allows you to customize:

- **Model Architecture**: Hidden layer sizes, dropout rate
- **Training Parameters**: Batch size, epochs, learning rate, scheduler
- **Quantization Settings**: Backend (fbgemm/qnnpack), observer type
- **Data Processing**: Normalization, augmentation
- **Logging**: TensorBoard, Weights & Biases integration

### Example Configuration

```yaml
model:
  input_size: 784
  hidden_sizes: [256, 128]
  output_size: 10
  dropout_rate: 0.2

training:
  batch_size: 64
  epochs: 10
  learning_rate: 0.001
  weight_decay: 1e-4

quantization:
  backend: "fbgemm"
  observer_type: "histogram"
  quantize_activations: true
  quantize_weights: true
```

## What is Quantization-Aware Training?

Quantization-Aware Training (QAT) simulates the effects of quantization during training, allowing models to learn to be robust to reduced precision (e.g., int8). Unlike post-training quantization, QAT yields higher accuracy and is better suited for edge devices, mobile, and IoT applications where low latency and small model size matter.

### Key Benefits:
- **Higher Accuracy**: Better performance compared to post-training quantization
- **Edge Device Ready**: Optimized for mobile and IoT deployments
- **Reduced Model Size**: Significant compression with minimal accuracy loss
- **Faster Inference**: Integer operations are faster than floating-point

## Web Dashboard Features

The Streamlit dashboard provides:

- **Overview**: Project description and recent experiments
- **Configuration**: Interactive parameter tuning
- **Training**: Start and monitor training sessions
- **Results**: Detailed training metrics and visualizations
- **Analytics**: Performance trends and insights

## Testing

Run the comprehensive test suite:

```bash
python test_qat.py
```

The test suite covers:
- Database operations
- Model architecture
- Training pipeline
- Integration tests

## Monitoring and Logging

### TensorBoard Integration
```bash
tensorboard --logdir logs/
```

### Database Queries
The SQLite database stores:
- Experiment metadata
- Training metrics per epoch
- Model performance statistics

### Visualizations
- Training/validation loss and accuracy curves
- Model size comparisons
- Performance analytics
- Backend comparisons

## 🔧 Advanced Usage

### Custom Model Architecture
Extend the `ModernQuantizedMLP` class to create custom architectures:

```python
class CustomQuantizedModel(ModernQuantizedMLP):
    def __init__(self):
        super().__init__(
            input_size=784,
            hidden_sizes=[512, 256, 128],
            output_size=10,
            dropout_rate=0.3
        )
```

### Custom Quantization Backend
Modify the quantization configuration for different hardware:

```yaml
quantization:
  backend: "qnnpack"  # For ARM processors
  # or
  backend: "fbgemm"   # For x86 processors
```

## Technical Details

### Model Architecture
- **Input**: 28×28 MNIST images (784 features)
- **Hidden Layers**: Configurable fully connected layers with BatchNorm and ReLU
- **Output**: 10 classes (digits 0-9)
- **Quantization**: QuantStub/DeQuantStub for simulation

### Training Process
1. **Data Preparation**: MNIST dataset with optional augmentation
2. **Model Setup**: Architecture definition and quantization preparation
3. **QAT Training**: Training with quantization simulation
4. **Conversion**: Convert to actual quantized model
5. **Evaluation**: Final performance assessment

### Database Schema
- **experiments**: Experiment metadata and final results
- **training_logs**: Per-epoch training metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent quantization APIs
- MNIST dataset creators
- Streamlit for the web framework
- The open-source community for various libraries

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples


# Quantization-Aware-Training
