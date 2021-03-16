import torch
import numpy as np

import matplotlib.pyplot as plt


def plot_loss(ckpt_path):
    which_model = ckpt_path[ckpt_path.rfind('/') + 1 : ].split('_')[0]
    checkpoint = torch.load(ckpt_path)
    losses = checkpoint['losses']    
    plt.plot(losses)
    plt.xlabel('Number of steps')
    plt.ylabel('Loss (cosine dist)')
    plt.title(f'{which_model} loss curve')
