import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm

# --- Core Logic Imports from our Project ---
from src.config import *
from src.dataset import EmojiDataset # Import the class, not the loader
from src.vq_vae.model import VQVAE 

def get_eval_loader():
    """
    Creates a DataLoader for passing all data.
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
    
    # Overwrite transform for simple evaluation
    eval_dataset.transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=VQ_VAE_BATCH_SIZE, # Use the training batch size
        shuffle=False, 
        num_workers=2
    )
    return eval_loader

def fix_model_stats(device):
    """
    Loads the trained model, re-computes BatchNorm stats, and saves it.
    """
    if not os.path.exists(VQ_VAE_BEST_MODEL_PATH):
        print(f"Error: Model file not found at {VQ_VAE_BEST_MODEL_PATH}")
        print("Please train the model first.")
        return

    print("--- Fixing VQ-VAE BatchNorm Statistics ---")
    
    # 1. Load Model
    print(f"Loading trained model from {VQ_VAE_BEST_MODEL_PATH}...")
    model = VQVAE(
        in_channels=3, hidden_dims=VQ_VAE_HIDDEN_DIMS, latent_dim=VQ_VAE_LATENT_DIM,
        num_embeddings=VQ_VAE_NUM_EMBEDDINGS, commitment_cost=VQ_VAE_COMMITMENT_COST,
        num_res_blocks=VQ_VAE_NUM_RES_BLOCKS, ema_decay=EMA_DECAY
    ).to(device)
    model.load_state_dict(torch.load(VQ_VAE_BEST_MODEL_PATH, map_location=device))
    
    # 2. Put model in TRAIN mode
    # This is critical. It tells BatchNorm layers to update their
    # running_mean and running_var stats.
    model.train()
    print("Model set to .train() mode to re-compute running statistics...")

    # 3. Load Data
    eval_loader = get_eval_loader()
    if eval_loader is None:
        return
    
    # 4. Pass all data through the model
    print(f"Running one pass over {len(eval_loader)} batches to update stats...")
    
    # We don't need gradients, just the forward pass
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Updating Stats"):
            x = batch.to(device)
            # Just run the forward pass. This updates the BatchNorm stats
            # because the model is in .train() mode.
            model(x)
            
    # 5. Save the "fixed" model
    # The model now has correct running_mean and running_var stats.
    print("Statistics re-computed.")
    print(f"Saving fixed model back to {VQ_VAE_BEST_MODEL_PATH}...")
    torch.save(model.state_dict(), VQ_VAE_BEST_MODEL_PATH)
    
    print("\n✅ Success! Your model file is now fixed.")
    print("You can now use model.eval() in all your scripts.")

if __name__ == "__main__":
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
        fix_model_stats(DEVICE)
    except Exception as e:
        print(f"\nAn error occurred while fixing stats: {e}")
        import traceback
        traceback.print_exc()

