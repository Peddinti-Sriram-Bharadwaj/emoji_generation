import os
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from src.config import IMAGE_SIZE, VQ_VAE_BATCH_SIZE, DATA_DIR, DATASET_PATH

class EmojiDataset(Dataset):
    def __init__(self, hf_dataset, image_size=64):
        self.dataset = hf_dataset
        # --- NEW: Added data augmentation ---
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)

def get_data_loader():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    try:
        dataset = load_dataset(DATASET_PATH, cache_dir=DATA_DIR, split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    print(f"Dataset size: {len(dataset)}")
    train_dataset = EmojiDataset(dataset, IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=VQ_VAE_BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Total batches: {len(train_loader)}")
    return train_loader
