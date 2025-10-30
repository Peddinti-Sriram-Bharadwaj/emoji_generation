# General Configuration
DEVICE = "mps"
DATASET_PATH = "valhalla/emoji-dataset"
DATA_DIR = "data/"
IMAGE_SIZE = 64

# VQ-VAE Configuration
VQ_VAE_BATCH_SIZE = 128 # Larger batch size helps codebook learning
VQ_VAE_NUM_EPOCHS = 200 # Train for longer with better stability
VQ_VAE_LEARNING_RATE = 2e-4 # Slightly higher LR with scheduler
VQ_VAE_HIDDEN_DIMS = [128, 256] # Simplified architecture
VQ_VAE_LATENT_DIM = 256
VQ_VAE_NUM_EMBEDDINGS = 512
VQ_VAE_COMMITMENT_COST = 0.25
VQ_VAE_NUM_RES_BLOCKS = 2
VQ_VAE_CHECKPOINT_PATH = "vqvae_checkpoint.pt"
VQ_VAE_BEST_MODEL_PATH = "vqvae_best.pt"

# --- NEW: Advanced Training Hyperparameters ---
# Codebook learning via Exponential Moving Average (EMA)
EMA_DECAY = 0.99
# When to reset unused codes (e.g., every 5 epochs)
CODEBOOK_RESET_INTERVAL = 5
# For learning rate scheduling
USE_LR_SCHEDULER = True
# For training stability
GRADIENT_CLIP_VAL = 1.0

# Transformer Prior Configuration (remains the same)
PRIOR_BATCH_SIZE = 64
PRIOR_NUM_EPOCHS = 100
PRIOR_LEARNING_RATE = 3e-4
PRIOR_EMBED_DIM = 512
PRIOR_NUM_HEADS = 8
PRIOR_NUM_LAYERS = 6
PRIOR_DROPOUT = 0.1
PRIOR_SEQ_LEN = 256
PRIOR_CHECKPOINT_PATH = "transformer_prior_checkpoint.pt"
PRIOR_BEST_MODEL_PATH = "transformer_prior_best.pt"
