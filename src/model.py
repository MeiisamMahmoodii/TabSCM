import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for row positions"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class TabPFNStyleEmbedding(nn.Module):
    """
    TabPFN-inspired embedding that uses a Transformer encoder over rows
    to generate rich, context-aware column embeddings.
    
    Instead of simple mean pooling, this:
    1. Encodes each (value, mask) pair with positional encoding
    2. Uses Transformer attention to capture distributional properties
    3. Summarizes with a learnable query vector
    """
    def __init__(self, embed_dim=192, n_heads=4, n_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Input projection: (value, mask) -> embed_dim
        self.input_proj = nn.Linear(2, embed_dim)
        
        # Positional encoding for row positions
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer encoder over rows
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.row_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Learnable query for column summarization
        self.column_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Cross-attention to summarize rows
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, m):
        # x: (batch, rows) - values for one column
        # m: (batch, rows) - intervention mask for one column
        batch_size, n_rows = x.shape
        
        # Combine value and mask
        combined = torch.stack([x, m], dim=-1)  # (batch, rows, 2)
        
        # Project to embedding space
        embeddings = self.input_proj(combined)  # (batch, rows, embed_dim)
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        
        # Encode rows with Transformer
        encoded = self.row_encoder(embeddings)  # (batch, rows, embed_dim)
        
        # Expand query for batch
        query = self.column_query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        
        # Cross-attention: query attends to encoded rows
        summary, _ = self.cross_attn(query, encoded, encoded)  # (batch, 1, embed_dim)
        
        # Squeeze and normalize
        output = self.output_norm(summary.squeeze(1))  # (batch, embed_dim)
        
        return output

class ZCIA_Transformer(nn.Module):
    def __init__(self, max_cols=128, embed_dim=192, n_heads=8, n_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Step 1: Embed each column (TabPFN-style)
        self.set_encoder = TabPFNStyleEmbedding(
            embed_dim=embed_dim,
            n_heads=4,  # For row-level attention
            n_layers=2   # For row-level encoding
        )
        
        # Step 2: Transformer to capture interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            batch_first=True,
            dim_feedforward=embed_dim*4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Step 3: Predict edges
        self.head = nn.Bilinear(embed_dim, embed_dim, 1)
        
        # Step 4: Auxiliary task - predict interventions
        self.intervention_head = nn.Linear(embed_dim, 1)

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
        
        # Predict Interventions (auxiliary task)
        intervention_logits = self.intervention_head(z).squeeze(-1)  # (batch, cols)
        
        return logits, intervention_logits
