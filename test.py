import torch
from model import VisualModel, VisionLanguageModel

checkpoint = torch.load('checkpoints/vision-only_100.pth')
print(checkpoint.keys())
model = VisualModel()
model.load_state_dict(checkpoint['state_dict'])
print(model)
#print(checkpoint['losses'])
