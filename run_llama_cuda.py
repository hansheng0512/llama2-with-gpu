import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import nvidia_smi
import gc
import os

# Set PyTorch memory allocator settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'


class GPUModelManager:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name

        # Initialize NVIDIA management
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        # Initialize CUDA settings
        self.setup_cuda_device()

        # Load model and tokenizer
        self.load_model_and_tokenizer()

    def setup_cuda_device(self):
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            print(f"CUDA available: {self.cuda_available}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            self.print_gpu_utilization()
        else:
            print("CUDA is not available. Using CPU.")

    def print_gpu_utilization(self):
        if self.cuda_available:
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
            print(f"GPU memory occupied: {info.used // 1024 ** 2} MB")
            print(f"GPU memory free: {info.free // 1024 ** 2} MB")
            print(f"GPU total memory: {info.total // 1024 ** 2} MB")

    def clear_gpu_memory(self):
        if self.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def load_model_and_tokenizer(self):
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set padding token to EOS token")

        print("Loading model with 4-bit quantization...")
        self.clear_gpu_memory()

        try:
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            model_kwargs = {
                "device_map": "auto",
                "quantization_config": quantization_config,
                "max_memory": {
                    0: "4GiB",
                    "cpu": "32GB"
                },
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(
            self,
            prompt: str,
            max_new_tokens: int = 50,
            temperature: float = 0.7,
            num_return_sequences: int = 1
    ) -> str:
        print("\nStarting generation...")
        self.print_gpu_utilization()

        try:
            # Process input text
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256  # Input context window size
            )

            if self.cuda_available:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            with torch.inference_mode(), autocast('cuda' if self.cuda_available else 'cpu'):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,  # Only using max_new_tokens
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    use_cache=True
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("\nGeneration completed!")
            self.print_gpu_utilization()

            return response

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.clear_gpu_memory()
                print("GPU out of memory. Cleared cache. Please try with smaller input/output lengths.")
            raise
        finally:
            self.clear_gpu_memory()

    def __del__(self):
        try:
            nvidia_smi.nvmlShutdown()
        except:
            pass


def main():
    manager = GPUModelManager()

    prompt = "Tell me a joke"
    print("\nPrompt:", prompt)

    print("\nGPU Memory Before Generation:")
    manager.print_gpu_utilization()

    try:
        response = manager.generate_response(
            prompt=prompt,
            max_new_tokens=50,  # Using only max_new_tokens
            temperature=0.7
        )
        print("\nResponse:", response)
    except Exception as e:
        print(f"Error during generation: {e}")
    finally:
        print("\nGPU Memory After Generation:")
        manager.print_gpu_utilization()


if __name__ == "__main__":
    main()