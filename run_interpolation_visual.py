import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.config import *
from src.dataset import get_data_loader
from src.vq_vae.model import ImprovedVQVAE

def create_visual_interpolation(model, img1, img2, num_steps=64, device='cpu'):
    """
    Creates a smooth visual morph by interpolating between the reconstructed images.
    """
    model.eval()
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    frames = []

    with torch.no_grad():
        # 1. Get the high-quality reconstructions of the start and end images
        recon_start, _, _, _ = model(img1)
        recon_end, _, _, _ = model(img2)

    print("Generating smooth visual interpolation (cross-fade)...")
    # 2. Linearly interpolate between the *pixel values* of the two reconstructions
    for alpha in np.linspace(0, 1, num_steps):
        interp_frame = (1 - alpha) * recon_start + alpha * recon_end
        frames.append(interp_frame.squeeze(0).cpu())
            
    return frames

def create_morph_animation(start_img, end_img, frames, filename="emoji_visual_morph.gif"):
    """
    Creates a GIF showing the start, end, and morphing emoji.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    
    # Display start image
    axes[0].imshow(start_img.permute(1, 2, 0))
    axes[0].set_title("Start Emoji")
    axes[0].axis('off')

    # Display end image
    axes[2].imshow(end_img.permute(1, 2, 0))
    axes[2].set_title("End Emoji")
    axes[2].axis('off')

    # Placeholder for the morphing image
    im = axes[1].imshow(frames[0].permute(1, 2, 0))
    axes[1].set_title("Morphing")
    axes[1].axis('off')
    
    fig.tight_layout()

    def update(frame_idx):
        im.set_array(frames[frame_idx].permute(1, 2, 0))
        return [im]

    print("Creating animation... This may take a moment.")
    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    anim.save(filename, writer='imagemagick', fps=20)
    plt.close(fig)
    print(f"✅ Visual morph animation saved to {filename}")

def main():
    device = torch.device(DEVICE)
    train_loader = get_data_loader()
    
    model = ImprovedVQVAE(in_channels=3, hidden_dims=VQ_VAE_HIDDEN_DIMS, latent_dim=VQ_VAE_LATENT_DIM, 
                          num_embeddings=VQ_VAE_NUM_EMBEDDINGS, commitment_cost=VQ_VAE_COMMITMENT_COST, 
                          num_res_blocks=VQ_VAE_NUM_RES_BLOCKS).to(device)
    model.load_state_dict(torch.load(VQ_VAE_BEST_MODEL_PATH, map_location=device))

    dataset = train_loader.dataset
    # Pick two nice, distinct emojis for a good visual
    img1, img2 = dataset[25], dataset[100]

    frames = create_visual_interpolation(model, img1, img2, device=device)
    create_morph_animation(img1, img2, frames)

if __name__ == "__main__":
    main()
