import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerPrior(nn.Module):
    def __init__(self, num_embeddings, embed_dim, num_heads, num_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(num_embeddings + 1, embed_dim)
        self.positional_embedding = nn.Embedding(seq_len + 1, embed_dim)
        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(embed_dim, num_embeddings)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(positions)
        x = token_embed + pos_embed
        x = self.transformer_encoder(x)
        return self.fc_out(x)
