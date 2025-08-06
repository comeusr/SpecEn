import torch
import sys

def check_cuda_setup():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Try a simple CUDA operation
        x = torch.rand(5, 3)
        print("CPU tensor:", x)
        x = x.cuda()
        print("GPU tensor:", x)
    else:
        print("CUDA is not available!")

if __name__ == "__main__":
    check_cuda_setup()

