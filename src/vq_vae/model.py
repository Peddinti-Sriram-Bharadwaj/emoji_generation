import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)

class Encoder(nn.Module):
    """
    The Encoder network that maps an input image to a lower-dimensional latent space.
    """
    def __init__(self, in_channels, hidden_dims, latent_dim, num_res_blocks):
        super().__init__()
        modules = []
        # Initial convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU()
        ))

        # Downsampling blocks with residual layers
        for i in range(len(hidden_dims) - 1):
            for _ in range(num_res_blocks):
                modules.append(ResidualBlock(hidden_dims[i]))
            modules.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU()
            ))
        
        # Final residual blocks
        for _ in range(num_res_blocks):
            modules.append(ResidualBlock(hidden_dims[-1]))
        
        # Final convolution to map to latent space
        modules.append(nn.Sequential(
            nn.Conv2d(hidden_dims[-1], latent_dim, 1),
            nn.BatchNorm2d(latent_dim)
        ))
        
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """
    The Decoder network that reconstructs an image from the quantized latent space.
    This version is corrected to ensure the output dimensions match the input.
    """
    def __init__(self, latent_dim, hidden_dims, out_channels, num_res_blocks):
        super().__init__()
        modules = []
        # Note: hidden_dims is the reversed list from the encoder, e.g., [256, 128]

        # Initial block to go from latent space to the first hidden dimension
        modules.append(nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dims[0], 1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU()
        ))
        for _ in range(num_res_blocks):
            modules.append(ResidualBlock(hidden_dims[0]))

        # First upsampling block (e.g., 16x16 -> 32x32)
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU()
            ))
            for _ in range(num_res_blocks):
                modules.append(ResidualBlock(hidden_dims[i+1]))
        
        # --- ARCHITECTURE FIX ---
        # Added the second, missing upsampling layer (e.g., 32x32 -> 64x64)
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1] // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[-1] // 2),
            nn.ReLU()
        ))
        # --- END FIX ---

        # Final convolution to map to the output (RGB) channels
        modules.append(nn.Sequential(
            nn.Conv2d(hidden_dims[-1] // 2, out_channels, 3, padding=1),
            nn.Sigmoid() # Use Sigmoid for output in [0, 1] range
        ))
        
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        return self.decoder(z)

class EMAQuantizer(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates for the codebook.
    This helps in stabilizing training and preventing codebook collapse.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Initialize codebook embeddings
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        
        # Buffers for EMA updates
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embedding_avg', torch.randn(num_embeddings, embedding_dim))
        
        # Buffer to track the usage of each code
        self.register_buffer('code_usage_counter', torch.zeros(num_embeddings, dtype=torch.long))

    def forward(self, z):
        # Reshape z from [B, C, H, W] to [B*H*W, C]
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_permuted.view(-1, self.embedding_dim)

        # Calculate distances between latents and embeddings
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(z_flattened, self.embedding.t()))

        # Find the closest embedding for each latent vector
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize the latents by replacing them with the closest embeddings
        quantized = torch.matmul(encodings, self.embedding).view(z_permuted.shape)

        # EMA Codebook Update (only during training)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(torch.sum(encodings, dim=0), alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(torch.matmul(encodings.t(), z_flattened), alpha=1 - self.decay)

            # Laplace smoothing to avoid zero counts
            n = torch.sum(self.cluster_size)
            cluster_size_smooth = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            
            # Update embeddings with the smoothed averages
            self.embedding.data.copy_(self.embedding_avg / cluster_size_smooth.unsqueeze(1))
            
            # Track code usage for resetting unused codes
            unique, counts = torch.unique(encoding_indices, return_counts=True)
            self.code_usage_counter[unique] += counts

        # Loss Calculation
        e_latent_loss = F.mse_loss(quantized.detach(), z_permuted)
        loss = self.commitment_cost * e_latent_loss

        # Straight-Through Estimator
        quantized = z_permuted + (quantized - z_permuted).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Perplexity Calculation (a measure of codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, encoding_indices.view(z.shape[0], z.shape[2], z.shape[3]), perplexity

    def reset_unused_codes(self, batch_latents):
        """
        Resets codebook vectors that have not been used to random vectors from the current batch.
        """
        unused_indices = torch.where(self.code_usage_counter == 0)[0]
        if len(unused_indices) == 0:
            return

        print(f"Resetting {len(unused_indices)} unused codebook vectors.")
        
        z_flattened = batch_latents.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        random_latents = z_flattened[torch.randint(0, z_flattened.size(0), (len(unused_indices),))]
        
        # Reinitialize the embeddings and EMA buffers for the unused codes
        self.embedding.data[unused_indices] = random_latents
        self.cluster_size.data[unused_indices] = 1.0
        self.embedding_avg.data[unused_indices] = random_latents
        
        # Reset the usage counter for the next tracking period
        self.code_usage_counter.zero_()


class VQVAE(nn.Module):
    """
    The complete VQ-VAE model, combining the Encoder, EMAQuantizer, and Decoder.
    """
    def __init__(self, in_channels, hidden_dims, latent_dim, num_embeddings, commitment_cost, num_res_blocks, ema_decay):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim, num_res_blocks)
        self.vq = EMAQuantizer(num_embeddings, latent_dim, commitment_cost, decay=ema_decay)
        # The hidden_dims for the decoder need to be reversed
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), in_channels, num_res_blocks)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, indices, perplexity = self.vq(z)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss, indices, perplexity
