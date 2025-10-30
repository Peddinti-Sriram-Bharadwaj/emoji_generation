import torch
from src.config import *
from src.dataset import get_data_loader
from src.vq_vae.model import ImprovedVQVAE
from src.evaluation import calculate_reconstruction_metrics # We only need this one metric function
from src.utils import plot_reconstructions

def main():
    # --- Setup ---
    device = torch.device(DEVICE)
    train_loader = get_data_loader()
    
    # --- Load VQ-VAE Model ---
    print("Loading trained VQ-VAE model...")
    vqvae_model = ImprovedVQVAE(
        in_channels=3, 
        hidden_dims=VQ_VAE_HIDDEN_DIMS, 
        latent_dim=VQ_VAE_LATENT_DIM, 
        num_embeddings=VQ_VAE_NUM_EMBEDDINGS, 
        commitment_cost=VQ_VAE_COMMITMENT_COST, 
        num_res_blocks=VQ_VAE_NUM_RES_BLOCKS
    ).to(device)
    
    # This loads the model you trained (either locally or on Colab)
    vqvae_model.load_state_dict(torch.load(VQ_VAE_BEST_MODEL_PATH, map_location=device))
    print("✅ VQ-VAE model loaded successfully.")

    # --- 1. Reconstruction Evaluation ---
    print("\n--- Evaluating Reconstructions ---")
    recon_metrics = calculate_reconstruction_metrics(vqvae_model, train_loader, device)
    print(f"\nReconstruction Metrics -> MSE: {recon_metrics['MSE']:.4f} | SSIM: {recon_metrics['SSIM']:.4f}")
    
    # --- 2. Visualize Reconstructions ---
    print("\nGenerating visualization of original vs. reconstructed images...")
    # Get one batch of images from the dataloader
    original_batch = next(iter(train_loader)).to(device)
    
    # Pass them through the model to get the reconstructions
    reconstructed_batch, _, _, _ = vqvae_model(original_batch)
    
    # Plot the comparison grid
    plot_reconstructions(original_batch, reconstructed_batch, n=8)
    print("✅ Visualization complete.")

if __name__ == "__main__":
    main()
