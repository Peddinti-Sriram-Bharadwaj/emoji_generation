import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.config import *

def train_epoch(model, loader, optimizer, device, grad_clip_val):
    model.train()
    total_loss, total_recon_loss, total_vq_loss, total_perplexity = 0, 0, 0, 0

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Pass the raw latents from the encoder to the code reset function if needed
        z = model.encoder(batch)
        quantized, vq_loss, _, perplexity = model.vq(z)
        reconstructed = model.decoder(quantized)
        
        recon_loss = F.mse_loss(reconstructed, batch)
        loss = recon_loss + vq_loss
        loss.backward()
        
        # --- NEW: Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()

    avg_perplexity = total_perplexity / len(loader)
    active_codes = torch.sum(model.vq.code_usage_counter > 0).item()
    
    return {
        "loss": total_loss / len(loader),
        "recon_loss": total_recon_loss / len(loader),
        "vq_loss": total_vq_loss / len(loader),
        "perplexity": avg_perplexity,
        "active_codes": active_codes
    }

def train_vq_vae(model, train_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=VQ_VAE_LEARNING_RATE)
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, VQ_VAE_NUM_EPOCHS, eta_min=1e-6)

    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(VQ_VAE_CHECKPOINT_PATH):
        # (Checkpoint loading logic remains the same)
        print("Resuming training...")
    else:
        print("Starting training from scratch.")

    for epoch in range(start_epoch, VQ_VAE_NUM_EPOCHS):
        metrics = train_epoch(model, train_loader, optimizer, device, GRADIENT_CLIP_VAL)
        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{VQ_VAE_NUM_EPOCHS} | Loss: {metrics['loss']:.4f} | "
            f"Recon: {metrics['recon_loss']:.4f} | VQ: {metrics['vq_loss']:.4f} | "
            f"Perplexity: {metrics['perplexity']:.2f} | "
            f"Active Codes: {metrics['active_codes']}/{VQ_VAE_NUM_EMBEDDINGS}"
        )

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save(model.state_dict(), VQ_VAE_BEST_MODEL_PATH)
            print(f"   -> New best model saved with loss: {best_loss:.4f}")

        # --- NEW: Periodically reset unused codes ---
        if (epoch + 1) % CODEBOOK_RESET_INTERVAL == 0:
            print("\n--- Performing Codebook Reset ---")
            # Get a sample batch to use for re-initialization
            sample_batch = next(iter(train_loader)).to(device)
            with torch.no_grad():
                sample_latents = model.encoder(sample_batch)
            model.vq.reset_unused_codes(sample_latents)
            print("---------------------------------\n")

        # (Checkpoint saving logic remains the same)

    torch.save(model.state_dict(), 'vqvae_final.pt')
    print("\n✅ Training complete. Final model saved.")
