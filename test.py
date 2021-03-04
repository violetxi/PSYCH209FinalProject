import torch

checkpoint = torch.load('checkpoints/vision-only_0.pth')
print(checkpoint['losses'])
