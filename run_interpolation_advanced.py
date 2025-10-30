import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from src.config import *
from src.dataset import get_data_loader
from src.vq_vae.model import ImprovedVQVAE # Make sure this is the corrected model file

def get_closest_indices(z_interp, codebook):
    """
    Finds the index of the closest codebook vector for each latent vector.
    """
    z_flattened = z_interp.view(-1, VQ_VAE_LATENT_DIM)
    distances = (torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(z_flattened, codebook.t()))
    return torch.argmin(distances, dim=1)

def interpolate_and_analyze(model, img1, img2, num_steps=32, device='cpu'):
    """
    Performs interpolation and logs latent space statistics at each step.
    """
    model.eval()
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    stats_log = []
    frames = []

    with torch.no_grad():
        # Get start and end quantized latents
        z1 = model.encoder(img1)
        z2 = model.encoder(img2)
        _, _, indices1, _ = model.vq(z1)
        _, _, indices2, _ = model.vq(z2)
        quantized1 = model.vq.embeddings(indices1.flatten()).view(z1.shape)
        quantized2 = model.vq.embeddings(indices2.flatten()).view(z2.shape)
        
        print("Generating and analyzing interpolation frames...")
        for alpha in tqdm(np.linspace(0, 1, num_steps)):
            # 1. Linearly interpolate in the continuous embedding space
            z_interp_continuous = (1 - alpha) * quantized1 + alpha * quantized2
            
            # 2. Find the actual closest codebook vectors for this interpolated point
            interp_indices = get_closest_indices(z_interp_continuous, model.vq.embeddings.weight)
            quantized_interp = model.vq.embeddings(interp_indices).view(z1.shape)

            # 3. Decode the result from the actual codebook path
            decoded_frame = model.decoder(quantized_interp)
            frames.append(decoded_frame.squeeze(0).cpu())
            
            # 4. Calculate and log statistics
            dist_to_start = torch.norm(quantized_interp - quantized1).item()
            dist_to_end = torch.norm(quantized_interp - quantized2).item()
            unique_codes = len(torch.unique(interp_indices))
            stats_log.append({
                'alpha': alpha,
                'dist_to_start': dist_to_start,
                'dist_to_end': dist_to_end,
                'unique_codes': unique_codes,
            })
            
    return frames, stats_log

def create_informative_animation(frames, stats, filename="interpolation_analysis.gif"):
    """
    Creates a GIF with the morphing emoji and plots of the latent stats.
    """
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Main plot for the emoji
    ax_img = fig.add_subplot(gs[:, 0])
    im = ax_img.imshow(frames[0].permute(1, 2, 0))
    ax_img.axis('off')
    ax_img.set_title("Interpolated Emoji")

    # Plot for latent distance
    ax_dist = fig.add_subplot(gs[0, 1])
    line_start, = ax_dist.plot([], [], 'b-', label='Dist to Start')
    line_end, = ax_dist.plot([], [], 'r-', label='Dist to End')
    ax_dist.set_xlim(0, 1)
    ax_dist.set_ylim(0, max(s['dist_to_start'] for s in stats) * 1.1)
    ax_dist.set_title("L2 Distance in Latent Space")
    ax_dist.set_xlabel("Interpolation Step (alpha)")
    ax_dist.set_ylabel("Distance")
    ax_dist.legend()
    ax_dist.grid(True)
    
    # Plot for codebook usage
    ax_codes = fig.add_subplot(gs[1, 1])
    line_codes, = ax_codes.plot([], [], 'g-')
    ax_codes.set_xlim(0, 1)
    ax_codes.set_ylim(0, VQ_VAE_NUM_EMBEDDINGS)
    ax_codes.set_title("Codebook Usage")
    ax_codes.set_xlabel("Interpolation Step (alpha)")
    ax_codes.set_ylabel("Unique Codes Used")
    ax_codes.grid(True)

    def update(frame_idx):
        # Update image
        im.set_array(frames[frame_idx].permute(1, 2, 0))
        
        # Update plots
        x_data = [s['alpha'] for s in stats[:frame_idx+1]]
        dist_start_data = [s['dist_to_start'] for s in stats[:frame_idx+1]]
        dist_end_data = [s['dist_to_end'] for s in stats[:frame_idx+1]]
        codes_data = [s['unique_codes'] for s in stats[:frame_idx+1]]
        
        line_start.set_data(x_data, dist_start_data)
        line_end.set_data(x_data, dist_end_data)
        line_codes.set_data(x_data, codes_data)
        
        return [im, line_start, line_end, line_codes]

    print("Creating animation... This may take a moment.")
    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    anim.save(filename, writer='imagemagick', fps=10)
    plt.close(fig)
    print(f"✅ Informative animation saved to {filename}")


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
    dataset = train_loader.dataset
    img1, img2 = dataset[25], dataset[50] # Pick two distinct emojis

    # --- Perform Interpolation and Analysis ---
    frames, stats_log = interpolate_and_analyze(model, img1, img2, num_steps=64, device=device)

    # --- Save Animation ---
    create_informative_animation(frames, stats_log)

if __name__ == "__main__":
    main()
