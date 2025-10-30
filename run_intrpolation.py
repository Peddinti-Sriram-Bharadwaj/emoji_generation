import torch
import numpy as np
import imageio
from tqdm import tqdm
from src.config import *
from src.dataset import get_data_loader
from src.vq_vae.model import ImprovedVQVAE
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def interpolate_latents(model, img1, img2, num_steps=16, device='cpu'):
    """
    Interpolates between the latent spaces of two images and decodes the results.
    """
    model.eval()
    
    # Ensure images have a batch dimension
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Encode both images to get their latent representations
        z1 = model.encoder(img1)
        z2 = model.encoder(img2)
        
        # 2. Quantize the latents to get the codebook vectors
        _, _, indices1, _ = model.vq(z1)
        _, _, indices2, _ = model.vq(z2)
        
        # Get the actual embedding vectors from the codebook
        quantized1 = model.vq.embeddings(indices1.flatten()).view(z1.shape)
        quantized2 = model.vq.embeddings(indices2.flatten()).view(z2.shape)

        frames = []
        # 3. Perform linear interpolation on the quantized embedding vectors
        for alpha in np.linspace(0, 1, num_steps):
            # Interpolate the quantized vectors
            z_interp = (1 - alpha) * quantized1 + alpha * quantized2
            
            # 4. Decode the interpolated latent vector
            decoded_frame = model.decoder(z_interp)
            frames.append(decoded_frame.squeeze(0).cpu())
            
    return frames

def save_animation(frames, filename="interpolation.gif", duration=150):
    """
    Saves a list of image tensors as a GIF.
    """
    # Convert tensors to numpy arrays suitable for imageio
    images = [(frame.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8) for frame in frames]
    
    # Add a pause at the end by duplicating the last frame
    images.extend([images[-1]] * 5)
    
    imageio.mimsave(filename, images, duration=duration, loop=0)
    print(f"✅ Animation saved to {filename}")

def main():
    # --- Setup ---
    device = torch.device(DEVICE)
    train_loader = get_data_loader()
    
    # --- Load Model ---
    print("Loading trained VQ-VAE model...")
    model = ImprovedVQVAE(in_channels=3, hidden_dims=VQ_VAE_HIDDEN_DIMS, latent_dim=VQ_VAE_LATENT_DIM, 
                          num_embeddings=VQ_VAE_NUM_EMBEDDINGS, commitment_cost=VQ_VAE_COMMITMENT_COST, 
                          num_res_blocks=VQ_VAE_NUM_RES_BLOCKS).to(device)
    model.load_state_dict(torch.load(VQ_VAE_BEST_MODEL_PATH, map_location=device))
    print("✅ Model loaded successfully.")

    # --- Select Images ---
    # Get a batch and pick two images for interpolation
    dataset = train_loader.dataset
    img1 = dataset[0]  # First emoji
    img2 = dataset[10] # A different emoji

    # Visualize the start and end images
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(img1.permute(1, 2, 0))
    axes[0].set_title("Start Image")
    axes[0].axis('off')
    axes[1].imshow(img2.permute(1, 2, 0))
    axes[1].set_title("End Image")
    axes[1].axis('off')
    plt.show()

    # --- Perform Interpolation ---
    print("\n--- Generating Interpolation Frames ---")
    interpolation_frames = interpolate_latents(model, img1, img2, num_steps=32, device=device)

    # --- Save Animation ---
    save_animation(interpolation_frames, filename="emoji_morph.gif", duration=100)

if __name__ == "__main__":
    main()
