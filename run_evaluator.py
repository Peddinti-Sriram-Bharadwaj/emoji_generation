import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image import FrechetInceptionDistance
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import argparse # Added for command-line arguments

# --- Core Logic Imports from our Project ---
from src.config import *
from src.dataset import EmojiDataset # Import the class, not the loader
from src.vq_vae.model import VQVAE # Use the correct VQVAE class

# --- Configuration ---
EVAL_BATCH_SIZE = 64
EVAL_LIMIT_BATCHES = 50 
TSNE_LIMIT_SAMPLES = 1000 
FID_FEATURE_DIM = 2048 

# --- Fast Mode Configuration ---
FAST_EVAL_LIMIT_BATCHES = 10
FAST_TSNE_LIMIT_SAMPLES = 500

# --- Output Files ---
RECONSTRUCTION_PLOT_PATH = "evaluation_reconstructions.png"
TSNE_PLOT_PATH = "evaluation_latent_space_tsne.png"

def get_eval_loader(batch_size):
    """
    Creates a DataLoader specifically for evaluation.
    - No data augmentation (only resize/tensor)
    - No shuffling
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    try:
        dataset_hf = load_dataset(DATASET_PATH, cache_dir=DATA_DIR, split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    
    eval_dataset = EmojiDataset(dataset_hf, IMAGE_SIZE)
    
    eval_dataset.transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    return eval_loader

def to_uint8(images):
    """Scales float images from [0, 1] to [0, 255] uint8."""
    return (images.clamp(0, 1) * 255).byte()

def run_evaluation(device, fast=False):
    """
    Main function to run all evaluations on the trained VQ-VAE model.
    Includes a 'fast' flag to run on a smaller subset for speed.
    """
    if not os.path.exists(VQ_VAE_BEST_MODEL_PATH):
        print(f"Error: Model file not found at {VQ_VAE_BEST_MODEL_PATH}")
        print("Please run fix_batchnorm_stats.py first.")
        return

    print("--- VQ-VAE Model Evaluation ---")
    
    # Set limits based on whether 'fast' mode is enabled
    if fast:
        print("--- RUNNING IN FAST MODE ---")
        limit_batches = FAST_EVAL_LIMIT_BATCHES
        limit_tsne = FAST_TSNE_LIMIT_SAMPLES
    else:
        print("--- RUNNING IN FULL EVALUATION MODE ---")
        limit_batches = EVAL_LIMIT_BATCHES
        limit_tsne = TSNE_LIMIT_SAMPLES
    
    # 1. Load Model
    print(f"Loading *fixed* model from {VQ_VAE_BEST_MODEL_PATH}...")
    model = VQVAE(
        in_channels=3, hidden_dims=VQ_VAE_HIDDEN_DIMS, latent_dim=VQ_VAE_LATENT_DIM,
        num_embeddings=VQ_VAE_NUM_EMBEDDINGS, commitment_cost=VQ_VAE_COMMITMENT_COST,
        num_res_blocks=VQ_VAE_NUM_RES_BLOCKS, ema_decay=EMA_DECAY
    ).to(device)
    model.load_state_dict(torch.load(VQ_VAE_BEST_MODEL_PATH, map_location=device))
    
    # This is now correct, assuming fix_batchnorm_stats.py was run
    model.eval()
    print("Model set to .eval() mode.")

    # 2. Load Data
    print("Loading evaluation dataset...")
    eval_loader = get_eval_loader(batch_size=EVAL_BATCH_SIZE)
    if eval_loader is None:
        return
    
    # 3. Initialize Metrics
    print("Initializing metrics on CPU...")
    metrics_device = torch.device("cpu")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(metrics_device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(metrics_device)
    fid_metric = FrechetInceptionDistance(feature=FID_FEATURE_DIM, normalize=True).to(metrics_device)
    
    all_mses = []
    all_perplexities = []
    all_indices = []
    
    print(f"Running metrics on {limit_batches or len(eval_loader)} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            if limit_batches and i >= limit_batches:
                break
            
            x = batch.to(device)
            x_hat, vq_loss, indices, perplexity = model(x)
            
            x_cpu = x.to(metrics_device)
            x_hat_cpu = x_hat.to(metrics_device)

            all_mses.append(F.mse_loss(x_hat_cpu, x_cpu).item())
            ssim_metric.update(x_hat_cpu, x_cpu)
            psnr_metric.update(x_hat_cpu, x_cpu)
            
            x_uint8 = to_uint8(x_cpu)
            x_hat_uint8 = to_uint8(x_hat_cpu)
            fid_metric.update(x_uint8, real=True)
            fid_metric.update(x_hat_uint8, real=False)
            
            all_perplexities.append(perplexity.item())
            all_indices.append(indices.cpu())
            
    # 4. Compute Final Metrics (on CPU)
    print("\n--- Quantitative Results ---")
    
    avg_mse = np.mean(all_mses)
    avg_ssim = ssim_metric.compute().item()
    avg_psnr = psnr_metric.compute().item()
    final_fid = fid_metric.compute().item()
    avg_perplexity = np.mean(all_perplexities)
    
    print(f"[Reconstruction Quality]")
    print(f"  Mean Squared Error (MSE): {avg_mse:.6f} (Lower is better)")
    print(f"  Peak Signal-to-Noise (PSNR): {avg_psnr:.2f} dB (Higher is better)")
    print(f"  Structural Similarity (SSIM): {avg_ssim:.4f} (Higher is better)")
    
    print(f"\n[Generative Quality Proxy (Reconstruction FID)]")
    print(f"  Fréchet Inception Distance (FID): {final_fid:.2f} (Lower is better)")
    print("  (Note: This is FID between originals and reconstructions, not from a prior)")

    all_indices = torch.cat(all_indices).flatten()
    unique_codes_used = len(torch.unique(all_indices))
    total_codes = model.vq.num_embeddings
    usage_percentage = (unique_codes_used / total_codes) * 100
    
    print(f"\n[Latent Space Analysis]")
    print(f"  Average Codebook Perplexity: {avg_perplexity:.2f}")
    print(f"  Codebook Usage: {unique_codes_used} / {total_codes} ({usage_percentage:.1f}%)")

    # 5. Run Visualizations
    print("\n--- Generating Visualizations ---")
    
    vis_loader = get_eval_loader(batch_size=EVAL_BATCH_SIZE)
    if vis_loader is None:
        return
    
    visualize_reconstructions(model, vis_loader, device)
    # Pass the correct sample limit (fast or full) to the t-SNE function
    visualize_latent_space(model, vis_loader, device, limit_tsne)

    print(f"\n✅ Evaluation complete. Check '{RECONSTRUCTION_PLOT_PATH}' and '{TSNE_PLOT_PATH}'.")

def visualize_reconstructions(model, dataloader, device, num_images=8):
    print(f"Generating reconstruction plot at '{RECONSTRUCTION_PLOT_PATH}'...")
    model.eval() 
    with torch.no_grad():
        x = next(iter(dataloader)).to(device)
        x_hat, _, _, _ = model(x)
    
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    x_hat = x_hat.cpu().clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4.5))
    fig.suptitle("Originals (Top) vs. Reconstructions (Bottom)", fontsize=16)
    
    for i in range(num_images):
        axes[0, i].imshow(x[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Original {i+1}")
        
        axes[1, i].imshow(x_hat[i])
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Recon {i+1}")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(RECONSTRUCTION_PLOT_PATH)
    plt.close(fig)

def visualize_latent_space(model, dataloader, device, num_samples):
    print(f"Generating t-SNE plot at '{TSNE_PLOT_PATH}'...")
    model.eval() 
    all_latents_flat = []
    
    print(f"  Gathering {num_samples} latents for t-SNE...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break
            
            x = batch.to(device)
            _, _, indices, _ = model(x)
            
            latents_flat = indices.cpu().view(indices.size(0), -1).numpy()
            all_latents_flat.append(latents_flat)
            
    latents_to_plot = np.concatenate(all_latents_flat, axis=0)[:num_samples]
    
    print(f"  Running t-SNE on {len(latents_to_plot)} samples (this may take a minute)...")
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, max_iter=1000)
    
    tsne_results = tsne.fit_transform(latents_to_plot)
    
    print("  Plotting results...")
    fig = plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, s=10)
    plt.title("t-SNE Visualization of VQ-VAE Latent Space (Codebook Indices)", fontsize=16)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.2)
    plt.savefig(TSNE_PLOT_PATH)
    plt.close(fig)

if __name__ == "__main__":
    # --- Add Argument Parser ---
    parser = argparse.ArgumentParser(description="Run VQ-VAE Evaluation Script")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a quick evaluation on a small subset of data."
    )
    args = parser.parse_args()
    # --- End Argument Parser ---

    # --- MPS Device Setup ---
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = torch.device("mps")
        print(f"✅ Using Apple MPS device: {DEVICE}")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"⚠️ MPS not available. Falling back to CUDA: {DEVICE}")
    else:
        DEVICE = torch.device("cpu")
        print(f"⚠️ MPS and CUDA not available. Falling back to CPU: {DEVICE}")
    # --- End Setup ---

    try:
        # Pass the 'fast' flag to the main function
        run_evaluation(DEVICE, fast=args.fast)
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()


