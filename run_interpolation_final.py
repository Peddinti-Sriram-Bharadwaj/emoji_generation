import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from src.config import *
from src.dataset import get_data_loader
from src.vq_vae.model import ImprovedVQVAE

def precompute_dataset_latents(model, dataloader, device):
    """
    Encodes the entire dataset to get a library of all latent representations.
    This is computationally expensive but needs to be done only once.
    """
    model.eval()
    all_latents = []
    print("Pre-computing latent representations for the entire dataset...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding dataset"):
            batch = batch.to(device)
            z = model.encoder(batch)
            # We will use the continuous latent space representation for distance comparison
            all_latents.append(z.cpu())
    return torch.cat(all_latents, dim=0)

def find_nearest_neighbor_frames(model, img1, img2, dataset_latents, full_dataset, num_steps, device):
    """
    Finds the nearest emoji in the dataset for each point in the latent interpolation.
    """
    model.eval()
    stats_log = []
    visual_frames = []
    
    with torch.no_grad():
        # Get start and end quantized latents
        z1 = model.encoder(img1.unsqueeze(0).to(device))
        z2 = model.encoder(img2.unsqueeze(0).to(device))
        quantized1 = model.vq(z1)[0] # Get the quantized output
        quantized2 = model.vq(z2)[0]
        
        print("\nFinding nearest neighbors along the latent path...")
        for alpha in tqdm(np.linspace(0, 1, num_steps), desc="Interpolating"):
            # 1. Interpolate in the continuous embedding space
            z_interp = (1 - alpha) * quantized1 + alpha * quantized2
            
            # 2. Find the nearest neighbor in the pre-computed dataset latents
            distances = torch.norm((dataset_latents.to(device) - z_interp).flatten(start_dim=1), dim=1)
            nearest_idx = torch.argmin(distances).item()
            
            # The visual frame is the actual emoji from the dataset
            visual_frames.append(full_dataset[nearest_idx])
            
            # 3. For stats, we still analyze the properties of the interpolated point itself
            _, _, interp_indices, _ = model.vq(z_interp)
            quantized_interp = model.vq.embeddings(interp_indices.flatten()).view(z1.shape)

            dist_to_start = torch.norm(quantized_interp - quantized1).item()
            dist_to_end = torch.norm(quantized_interp - quantized2).item()
            unique_codes = len(torch.unique(interp_indices))
            stats_log.append({
                'alpha': alpha, 'dist_to_start': dist_to_start,
                'dist_to_end': dist_to_end, 'unique_codes': unique_codes,
            })
            
    return visual_frames, stats_log

def create_final_animation(start_img, end_img, visual_frames, stats, filename="nearest_neighbor_walk.gif"):
    """Creates a GIF showing the nearest neighbor walk and analysis graphs."""
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Top Row: Visuals
    ax_start = fig.add_subplot(gs[0, 0])
    ax_start.imshow(start_img.permute(1, 2, 0))
    ax_start.set_title("Start Emoji"); ax_start.axis('off')

    ax_morph = fig.add_subplot(gs[0, 1])
    im = ax_morph.imshow(visual_frames[0].permute(1, 2, 0))
    ax_morph.set_title("Nearest Dataset Emoji"); ax_morph.axis('off')

    ax_end = fig.add_subplot(gs[0, 2])
    ax_end.imshow(end_img.permute(1, 2, 0))
    ax_end.set_title("End Emoji"); ax_end.axis('off')

    # Bottom Row: Graphs
    ax_dist = fig.add_subplot(gs[1, :2])
    line_start, = ax_dist.plot([], [], 'b-', label='Dist to Start')
    line_end, = ax_dist.plot([], [], 'r-', label='Dist to End')
    ax_dist.set_xlim(0, 1); ax_dist.set_ylim(0, max(s['dist_to_start'] for s in stats) * 1.1)
    ax_dist.set_title("Analysis of Latent Space Path"); ax_dist.set_xlabel("Interpolation Step (alpha)"); ax_dist.set_ylabel("L2 Distance")
    ax_dist.legend(); ax_dist.grid(True)
    
    ax_codes = fig.add_subplot(gs[1, 2])
    line_codes, = ax_codes.plot([], [], 'g-', label='Unique Codes')
    ax_codes.set_xlim(0, 1); ax_codes.set_ylim(0, max(s['unique_codes'] for s in stats) * 1.1)
    ax_codes.set_xlabel("Interpolation Step (alpha)"); ax_codes.set_ylabel("Unique Codes Used")
    ax_codes.legend(); ax_codes.grid(True)

    def update(frame_idx):
        im.set_array(visual_frames[frame_idx].permute(1, 2, 0))
        x_data = [s['alpha'] for s in stats[:frame_idx+1]]
        line_start.set_data(x_data, [s['dist_to_start'] for s in stats[:frame_idx+1]])
        line_end.set_data(x_data, [s['dist_to_end'] for s in stats[:frame_idx+1]])
        line_codes.set_data(x_data, [s['unique_codes'] for s in stats[:frame_idx+1]])
        return [im, line_start, line_end, line_codes]

    print("\nCreating final animation... This may take a moment.")
    anim = FuncAnimation(fig, update, frames=len(visual_frames), blit=True)
    anim.save(filename, writer='imagemagick', fps=10)
    plt.close(fig)
    print(f"✅ Nearest Neighbor Walk animation saved to {filename}")

def main():
    device = torch.device(DEVICE)
    # Use a non-shuffling dataloader for pre-computation to keep order consistent
    train_loader = get_data_loader()
    
    model = ImprovedVQVAE(in_channels=3, hidden_dims=VQ_VAE_HIDDEN_DIMS, latent_dim=VQ_VAE_LATENT_DIM, 
                          num_embeddings=VQ_VAE_NUM_EMBEDDINGS, commitment_cost=VQ_VAE_COMMITMENT_COST, 
                          num_res_blocks=VQ_VAE_NUM_RES_BLOCKS).to(device)
    model.load_state_dict(torch.load(VQ_VAE_BEST_MODEL_PATH, map_location=device))

    # This is a one-time, upfront cost. It can take a minute.
    dataset_latents = precompute_dataset_latents(model, train_loader, device)

    dataset = train_loader.dataset
    img1, img2 = dataset[25], dataset[100]
    num_steps = 64

    # Perform the nearest neighbor search
    visual_frames, latent_stats = find_nearest_neighbor_frames(model, img1, img2, dataset_latents, dataset, num_steps, device)

    # --- Print Terminal Stats ---
    print("\n--- Latent Path Statistics Summary ---")
    avg_codes_used = np.mean([s['unique_codes'] for s in latent_stats])
    max_dist = max(s['dist_to_start'] for s in latent_stats)
    print(f"Total Interpolation Steps: {num_steps}")
    print(f"Maximum L2 Distance in Latent Space: {max_dist:.2f}")
    print(f"Average Unique Codebook Vectors per Step: {avg_codes_used:.2f}")
    print("-" * 35)

    # --- Create Animation ---
    create_final_animation(img1, img2, visual_frames, latent_stats)

if __name__ == "__main__":
    main()
