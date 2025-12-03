import torch

if torch.cuda.is_available():
    print("CUDA is available! You can use the GPU.")
else:
    print("CUDA is NOT available. Using CPU.")