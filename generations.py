import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image # Use this for direct saving
import datetime

# --- Core Logic Imports from our Project ---
from src.config import *
from src.dataset import EmojiDataset # Import the class
from src.vq_vae.model import VQVAE # Use the correct VQVAE class

# --- Configuration ---
NUM_IMAGES_TO_SHOW = 8
BATCH_SIZE = NUM_IMAGES_TO_SHOW
ORIGINALS_FILENAME = "originals.png"
RECONS_FILENAME = "reconstructions.png"

def get_eval_loader(batch_size):
    """
    Creates a DataLoader specifically for evaluation (no augmentations/shuffle).
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    try:
        dataset_hf = load_dataset(DATASET_PATH, cache_dir=DATA_DIR, split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    
    eval_dataset = EmojiDataset(dataset_hf, IMAGE_SIZE)
    
    # Overwrite transform for simple evaluation
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

def save_reconstruction_grid(device):
    """
    Loads the model, gets one batch, and saves comparison grids to disk.
    """
    if not os.path.exists(VQ_VAE_BEST_MODEL_PATH):
        print(f"Error: Model file not found at {VQ_VAE_BEST_MODEL_PATH}")
        print("Please train the model first.")
        return

    print("--- Saving VQ-VAE Reconstruction Grids ---")
    
    # 1. Load Model
    try:
        mtime = os.path.getmtime(VQ_VAE_BEST_MODEL_PATH)
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Loading model from {VQ_VAE_BEST_MODEL_PATH}")
        print(f"       (File last modified: {mtime_str})")
    except Exception as e:
        print(f"Loading model from {VQ_VAE_BEST_MODEL_PATH} (Could not get timestamp: {e})")
        
    model = VQVAE(
        in_channels=3, hidden_dims=VQ_VAE_HIDDEN_DIMS, latent_dim=VQ_VAE_LATENT_DIM,
        num_embeddings=VQ_VAE_NUM_EMBEDDINGS, commitment_cost=VQ_VAE_COMMITMENT_COST,
        num_res_blocks=VQ_VAE_NUM_RES_BLOCKS, ema_decay=EMA_DECAY
    ).to(device) # Send model to MPS/device
    model.load_state_dict(torch.load(VQ_VAE_BEST_MODEL_PATH, map_location=device))
    
    # --- THIS IS THE FIX ---
    # We call model.train() instead of model.eval()
    # This forces BatchNorm to use batch statistics instead of the (likely corrupt)
    # running statistics, which is what it did during training.
    model.train()
    print("Model set to .train() mode to bypass BatchNorm running_stats.")
    # --- END FIX ---


    # 2. Load Data
    print("Loading evaluation dataset...")
    eval_loader = get_eval_loader(batch_size=BATCH_SIZE)
    if eval_loader is None:
        return
    
    # 3. Get one batch and run inference
    print("Generating reconstructions...")
    with torch.no_grad():
        # Get one batch of original images
        x = next(iter(eval_loader)).to(device)
        # Get the model's reconstructions
        x_hat, _, _, _ = model(x)
    
    # 4. Clamp outputs just in case (to ensure valid [0,1] range)
    x_hat = x_hat.clamp(0.0, 1.0)
    
    # 5. Save the images directly to files
    save_image(x, ORIGINALS_FILENAME, nrow=NUM_IMAGES_TO_SHOW)
    save_image(x_hat, RECONS_FILENAME, nrow=NUM_IMAGES_TO_SHOW)
    
    print(f"\n✅ Success! Image grids saved to:")
    print(f"  - {ORIGINALS_FILENAME}")
    print(f"  - {RECONS_FILENAME}")
    print("\nPlease open these files to check the reconstructions.")

if __name__ == "__main__":
    # --- MPS Device Setup ---
    # Prioritize MPS since you are on macOS
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
        save_reconstruction_grid(DEVICE)
    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()


