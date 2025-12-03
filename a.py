import torch
print(torch.cuda.is_available())
# 利用可能なGPUの数を表示
print(torch.cuda.device_count())