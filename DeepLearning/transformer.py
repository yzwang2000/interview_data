import torch
import torch.nn as nn
import torch.nn.functional as F

# 写这部分代码的时候要注意几个点, 首先就是先 shortcut add 以后, 再 layerNorm。
# dropout 的地方是 MHA 中 softmax 之后, FF 中 Relu 之后。shortcut add 前有 dropout。
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), v)
        return attn_output

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Apply attention on all the projected vectors in batch
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear layer
        output = self.out_linear(attn_output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        attn_output = self.self_attn(src, src_mask)
        src = src + self.dropout(attn_output)
        src = self.layernorm1(src)

        # Feedforward network
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.layernorm2(src)

        return src

# Example usage:
d_model = 512
nhead = 8
transformer_block = TransformerBlock(d_model, nhead)

# Dummy input
src = torch.rand(32, 10, d_model)  # (batch_size, sequence_length, d_model)

output = transformer_block(src)
print(output.shape)  # Should print: torch.Size([32, 10, 512])