import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

def plot_reconstructions(original, reconstructed, n=8):
    original = original.cpu().numpy()
    reconstructed = reconstructed.detach().cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(12, 3))
    for i in range(n):
        axes[0, i].imshow(np.transpose(original[i], (1, 2, 0)))
        axes[0, i].axis('off')
        axes[0, 0].set_title('Original')

        axes[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        axes[1, i].axis('off')
        axes[1, 0].set_title('Reconstructed')
    plt.show()

def plot_generated(images, n=8):
    images = images.detach().cpu()
    grid = make_grid(images[:n], nrow=n)
    plt.figure(figsize=(12, 3))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Generated Samples')
    plt.show()
