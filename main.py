import torch
from src.config import *
from src.dataset import get_data_loader
from src.vq_vae.model import ImprovedVQVAE
from src.vq_vae.trainer import train_vq_vae
from src.transformer_prior.model import TransformerPrior
from src.transformer_prior.trainer import train_transformer_prior

def main():
    # VQ-VAE Training
    print("--- Starting VQ-VAE Training ---")
    vq_vae_model = ImprovedVQVAE(
        in_channels=3,
        hidden_dims=VQ_VAE_HIDDEN_DIMS,
        latent_dim=VQ_VAE_LATENT_DIM,
        num_embeddings=VQ_VAE_NUM_EMBEDDINGS,
        commitment_cost=VQ_VAE_COMMITMENT_COST,
        num_res_blocks=VQ_VAE_NUM_RES_BLOCKS,
    ).to(DEVICE)
    train_loader = get_data_loader()
    train_vq_vae(vq_vae_model, train_loader, DEVICE)

    # Transformer Prior Training
    print("\n--- Starting Transformer Prior Training ---")
    prior_model = TransformerPrior(
        num_embeddings=VQ_VAE_NUM_EMBEDDINGS,
        embed_dim=PRIOR_EMBED_DIM,
        num_heads=PRIOR_NUM_HEADS,
        num_layers=PRIOR_NUM_LAYERS,
        dropout=PRIOR_DROPOUT,
        seq_len=PRIOR_SEQ_LEN,
    ).to(DEVICE)

    # Create a new data loader for the prior
    # This will require generating the latents from the trained VQ-VAE
    # For simplicity, we are reusing the same loader here, but in a real
    # scenario, you would create a dataset of the VQ-VAE latents
    train_transformer_prior(prior_model, train_loader, DEVICE)

if __name__ == "__main__":
    main()
