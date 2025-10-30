import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
# CORRECT
from torchmetrics.functional.regression import mean_squared_error
from tqdm import tqdm

def calculate_reconstruction_metrics(model, dataloader, device):
    """
    Calculates reconstruction metrics (MSE, SSIM) for a VQ-VAE model.
    """
    model.eval()
    total_mse = 0
    
    # Initialize SSIM metric from torchmetrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calculating Reconstruction Metrics")):
            batch = batch.to(device)
            reconstructed, _, _, _ = model(batch)
            
            # Clamp values to [0, 1] for metric calculation
            batch_clamped = torch.clamp(batch, 0, 1)
            reconstructed_clamped = torch.clamp(reconstructed, 0, 1)

            total_mse += mean_squared_error(reconstructed_clamped, batch_clamped).item()
            ssim.update(reconstructed_clamped, batch_clamped)

            # Limit to 50 batches to speed up evaluation
            if i >= 50:
                break

    avg_mse = total_mse / (i + 1)
    final_ssim = ssim.compute().item()
    
    return {"MSE": avg_mse, "SSIM": final_ssim}


def calculate_fid(real_images_loader, generated_images_tensor, device):
    """
    Calculates the Fréchet Inception Distance (FID) between real and generated images.
    """
    # Note: FID expects images in range [0, 255] and as uint8
    generated_images_tensor = (generated_images_tensor * 255).byte()

    fid = FrechetInceptionDistance(feature=64).to(device) # Using inception features from layer 64 for speed
    
    # Update FID with real images
    print("Processing real images for FID...")
    for i, batch in enumerate(tqdm(real_images_loader, desc="FID - Real Batches")):
        batch = (batch.to(device) * 255).byte()
        fid.update(batch, real=True)
        # Limit to 50 batches
        if i >= 50:
            break
            
    # Update FID with generated images
    print("Processing generated images for FID...")
    fid.update(generated_images_tensor.to(device), real=False)

    return {"FID": fid.compute().item()}
