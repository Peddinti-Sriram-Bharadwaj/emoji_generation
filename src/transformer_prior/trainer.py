import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.config import (
    PRIOR_CHECKPOINT_PATH,
    PRIOR_BEST_MODEL_PATH,
    PRIOR_NUM_EPOCHS,
    PRIOR_LEARNING_RATE,
)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training Prior"):
        batch = batch.to(device)
        optimizer.zero_grad()
        targets = batch.clone()
        outputs = model(batch)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def train_transformer_prior(model, train_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRIOR_LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PRIOR_NUM_EPOCHS, eta_min=1e-6)

    start_epoch = 0
    best_loss = float('inf')
    train_losses = []

    if os.path.exists(PRIOR_CHECKPOINT_PATH):
        checkpoint = torch.load(PRIOR_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        train_losses = checkpoint['train_losses']
        print(f"✅ Checkpoint found. Resuming training from epoch {start_epoch}.")
    else:
        print("✅ No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, PRIOR_NUM_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        train_losses.append(loss)

        print(f"Epoch {epoch+1}/{PRIOR_NUM_EPOCHS} | Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), PRIOR_BEST_MODEL_PATH)
            print(f"   -> New best model saved with loss: {best_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses,
            }, PRIOR_CHECKPOINT_PATH)
            print(f"   -> Checkpoint saved at epoch {epoch+1}")

    torch.save(model.state_dict(), 'transformer_prior_final.pt')
    print("\n✅ Training complete. Final model saved.")
