import torch
import torch.nn as nn

class SetEmbedding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.element_encoder = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, m):
        # Stack feature and mask: (Batch, Rows, 2)
        combined = torch.stack([x, m], dim=-1)
        # Encode & Pool
        encodings = self.element_encoder(combined)
        pooled = torch.mean(encodings, dim=1)
        return self.post_pool_mlp(pooled)

class ZCIA_Transformer(nn.Module):
    def __init__(self, max_cols=128, embed_dim=128, n_heads=4, n_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.set_encoder = SetEmbedding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            batch_first=True,
            dim_feedforward=embed_dim*4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Bilinear(embed_dim, embed_dim, 1)

    def forward(self, x, m, pad_mask):
        batch_size, n_rows, n_cols = x.shape
        
        # Flatten batch and cols to treat them as a list of sets
        x_flat = x.reshape(-1, n_rows) 
        m_flat = m.reshape(-1, n_rows)
        
        # Encode Sets
        col_embeddings_flat = self.set_encoder(x_flat, m_flat)
        col_embeddings = col_embeddings_flat.reshape(batch_size, n_cols, self.embed_dim)
        
        # Transformer Attention
        z = self.transformer(col_embeddings, src_key_padding_mask=pad_mask)
        
        # Predict Edges
        z_i = z.unsqueeze(2).expand(-1, -1, n_cols, -1)
        z_j = z.unsqueeze(1).expand(-1, n_cols, -1, -1)
        logits = self.head(z_i, z_j).squeeze(-1)
        
        return logits
