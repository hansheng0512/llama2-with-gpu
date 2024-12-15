# Llama GPU Example

A sample Python implementation demonstrating how to run Llama language models on GPU with memory optimizations and 4-bit quantization. This code provides a reference implementation for loading and running Llama models efficiently with PyTorch.

## Overview

This example shows how to:
- Load and run Llama models with GPU optimization
- Implement 4-bit quantization for memory efficiency
- Monitor and manage GPU memory usage
- Handle common GPU-related issues
- Generate text responses with customizable parameters

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- nvidia-smi
- bitsandbytes
- CUDA-capable GPU

## Key Features Demonstrated

- 4-bit quantization using BitsAndBytes
- GPU memory monitoring and management
- Automatic memory cleanup
- Error handling for common GPU issues
- Configurable generation parameters

## Usage Example

```python
from gpu_model_manager import GPUModelManager

# Initialize with Llama-2-7b-chat
manager = GPUModelManager()

# Generate text
response = manager.generate_response(
    prompt="Tell me a joke",
    max_new_tokens=50,
    temperature=0.7
)
print(response)
```

## Implementation Details

### Memory Optimization

The code implements several memory-saving techniques:
- 4-bit quantization configuration
- Automatic cache clearing
- Memory monitoring
- Configurable memory limits for GPU and CPU

### Model Configuration

Default settings used in this example:
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_kwargs = {
    "device_map": "auto",
    "max_memory": {
        0: "4GiB",
        "cpu": "32GB"
    }
}
```

### Key Methods Explained

- `setup_cuda_device()`: Checks CUDA availability and initializes settings
- `print_gpu_utilization()`: Monitors GPU memory usage
- `clear_gpu_memory()`: Implements memory cleanup
- `generate_response()`: Handles text generation with error handling

## Customization

You can modify the following parameters:
- Model name/path
- Memory limits
- Generation parameters (temperature, max tokens, etc.)
- Quantization settings

## Notes

- This is a reference implementation and may need adjustments based on your specific hardware and requirements
- Memory limits should be adjusted based on your GPU capabilities
- The code includes basic error handling that you might want to enhance for production use

## Contributing

This is a sample implementation. Feel free to:
- Suggest improvements
- Adapt the code for your needs
- Share optimizations

## License

[Insert your chosen license here]