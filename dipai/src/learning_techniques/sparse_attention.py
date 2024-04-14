import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_length=512, adaptive=True):
        super(SparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_length = max_length
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(self.head_dim * self.num_heads, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_length, 1, embed_dim))
        self.key_query_mapping = nn.Linear(self.head_dim, self.head_dim)
        if self.adaptive:
            self.adapt_sparsity = nn.Linear(self.head_dim * self.num_heads * 3, 1)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length_q, _ = query.size()
        _, seq_length_kv, _ = key.size()
        target_seq_length = min(seq_length_q, seq_length_kv)
        #print(f"Batch size: {batch_size}, seq_length_q: {seq_length_q}, seq_length_kv: {seq_length_kv}, target_seq_length: {target_seq_length}")
        query = query[:, :target_seq_length, :]
        key = key[:, :target_seq_length, :]
        value = value[:, :target_seq_length, :]
        pos_encoding_q = self.positional_encoding[:target_seq_length, :, :].repeat(1, batch_size, 1).permute(1, 0, 2)
        pos_encoding_kv = pos_encoding_q 
        assert query.size(-1) == self.embed_dim, f"Query tensor's last dimension should be {self.embed_dim}, but got {query.size(-1)}"
        q = self.query(query) + pos_encoding_q
        k = self.key(key) + pos_encoding_kv
        v = self.value(value) + pos_encoding_kv
        #print(f"Before reshape - Q shape: {query.shape}, K shape: {key.shape}, V shape: {value.shape}")
        if target_seq_length == 0:
            print("Warning: Encountered zero-length sequence. Returning zeros.")
            return torch.zeros(batch_size, seq_length_q, self.embed_dim, device=query.device)
        q = q.view(batch_size, target_seq_length, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.view(batch_size, target_seq_length, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch_size, target_seq_length, self.num_heads, -1).permute(0, 2, 1, 3)
        k_mapped = self.key_query_mapping(k)
        sparsity_factor = torch.tensor(0.1, device=query.device) 
        seq_length = min(query.size(1), key.size(1))
        top_k_value = int(seq_length * sparsity_factor.item()) 
        top_k = max(min(top_k_value, seq_length), 1)
        new_dim = self.embed_dim
        concatenated = torch.cat([q, k, v], dim=-1)
        concatenated = concatenated.view(batch_size, target_seq_length, -1)
        if self.adaptive:
            sparsity_logits = self.adapt_sparsity(concatenated).squeeze(-1)
        scores = torch.matmul(q, k_mapped.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        topk_scores, indices = scores.topk(k=top_k, dim=-1, largest=True, sorted=False)
        topk_scores = F.softmax(topk_scores, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attn, v)
        combined_heads = weighted_values.permute(0, 2, 1, 3).contiguous().view(batch_size, target_seq_length, new_dim)
        output = self.out(combined_heads)
        return output