import torch.nn as nn


# Transformer Block for MTLFormer
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention
        attn_output, _ = self.attention(x, x, x)
        # Add & Norm
        x = self.norm1(attn_output + x)
        # Feed Forward Network
        ffn_output = self.ffn(x)
        # Add & Norm
        out = self.norm2(ffn_output + x)
        return out


# Multi-task Learning Transformer model (MTLFormer)
class MTLFormer(nn.Module):
    def __init__(self, embed_size, heads, dropout, num_classes):
        super(MTLFormer, self).__init__()
        # Embedding layer
        self.transformer = TransformerBlock(embed_size, heads, dropout)
        # Task-specific output heads
        self.fc_activity = nn.Linear(embed_size, num_classes)  # Next Activity (classification)
        self.fc_time = nn.Linear(embed_size, 1)  # Next Event Time (regression)
        self.fc_remaining = nn.Linear(embed_size, 1)  # Remaining Time (regression)

    def forward(self, x):
        # Pass through transformer block
        x = self.transformer(x)
        # Task-specific outputs
        next_activity = self.fc_activity(x)
        next_time = self.fc_time(x)
        remaining_time = self.fc_remaining(x)
        return next_activity, next_time, remaining_time
