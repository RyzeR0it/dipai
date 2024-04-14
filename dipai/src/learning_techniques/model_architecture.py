import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sparse_attention import SparseAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out(context)
        return output

class Feedforward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = Feedforward(embed_dim, ff_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        src2 = self.attention(src)
        src = self.layernorm1(src + self.dropout(src2))
        src2 = self.ffn(src)
        src = self.layernorm2(src + self.dropout(src2))
        return src
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = SparseAttention(embed_dim, num_heads, max_length=512, adaptive=True)
        self.cross_attention = SparseAttention(embed_dim, num_heads, max_length=512, adaptive=True)
        self.feed_forward = Feedforward(embed_dim, ff_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attention(query=tgt, key=tgt, value=tgt, mask=tgt_mask)
        max_batch_size = max(tgt2.size(0), memory.size(0))
        if tgt2.size(0) != max_batch_size:
            padding_size = (max_batch_size - tgt2.size(0),) + tgt2.shape[1:]
            padding = torch.zeros(padding_size, dtype=tgt2.dtype, device=tgt2.device)
            tgt2 = torch.cat([tgt2, padding], dim=0)
            if tgt_mask is not None:
                tgt_mask_padding = torch.zeros(padding_size[:-1] + (tgt_mask.shape[-1],), dtype=tgt_mask.dtype, device=tgt_mask.device)
                tgt_mask = torch.cat([tgt_mask, tgt_mask_padding], dim=0)
        if memory.size(0) != max_batch_size:
            padding_size = (max_batch_size - memory.size(0),) + memory.shape[1:]
            padding = torch.zeros(padding_size, dtype=memory.dtype, device=memory.device)
            memory = torch.cat([memory, padding], dim=0)
            if memory_mask is not None:
                memory_mask_padding = torch.zeros(padding_size[:-1] + (memory_mask.shape[-1],), dtype=memory_mask.dtype, device=memory_mask.device)
                memory_mask = torch.cat([memory_mask, memory_mask_padding], dim=0)
        tgt2 = self.cross_attention(query=tgt2, key=memory, value=memory, mask=memory_mask)
        tgt2 = self.feed_forward(tgt2)
        return tgt2

class EfficientTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout=0.1, max_length=512, adaptive=True):
        super(EfficientTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EfficientTransformerLayer(embed_dim, num_heads, ff_dim, dropout, max_length, adaptive)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.dropout(x)

class EfficientTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, max_length=512, adaptive=True):
        super(EfficientTransformerLayer, self).__init__()
        self.sparse_attention = SparseAttention(embed_dim, num_heads, max_length, adaptive)
        self.feed_forward = Feedforward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None): 
        attn_output = self.sparse_attention(query=x, key=x, value=x, mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, src_vocab_size, tgt_vocab_size, embed_dim):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)
    
    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.output_layer(output)
        return output
    
class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, heads, ff_dim, dropout_rate=0.1):
        super(BaseModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = EfficientTransformerEncoder(embed_dim, num_layers, heads, ff_dim, dropout_rate)
        
    def forward(self, x):
        if not torch.is_tensor(x) or not x.dtype == torch.int64:
            print(f"Unexpected input type or dtype: {type(x)}, {x.dtype}. Attempting to correct.")
            x = x.to(torch.int64)
        if (x >= self.vocab_size).any() or (x < 0).any():
            raise ValueError("Input tensor contains out-of-range indices for embedding.")
        x = self.embedding(x)
        x = self.encoder(x)
        return x

class GeneralModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, heads, ff_dim, num_entities, num_intents, dropout_rate=0.1):
        super(GeneralModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.base_model = BaseModel(vocab_size, embed_dim, num_layers, heads, ff_dim, dropout_rate)
        self.sentiment_analysis = nn.Linear(embed_dim, 3)
        self.dialogue_contextualizer = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.entity_recognition_head = nn.Linear(embed_dim, num_entities)
        self.intent_classification_head = nn.Linear(embed_dim, num_intents)
        self.encoder = EfficientTransformerEncoder(embed_dim, num_layers, heads, ff_dim, dropout_rate, max_length=512, adaptive=True)
        self.decoder = TransformerDecoderLayer(embed_dim, heads, ff_dim, dropout_rate)
        self.seq2seq_output_layer = nn.Linear(embed_dim, vocab_size) 
        self.seq2seq_mode = True

    def _ensure_int_indices(self, tensor, context='unknown'):
        """Ensure tensor is of type torch.int64, correcting if necessary."""
        if not torch.is_tensor(tensor):
            print(f"[{context}] Non-tensor input detected. Attempting to convert.")
            tensor = torch.tensor(tensor)
        if tensor.dtype != torch.int64:
            print(f"[{context}] Unexpected dtype: {tensor.dtype}. Correcting to torch.int64.")
            tensor = tensor.to(torch.int64)
        
        return tensor

    def _check_indices(self, tensor, context='unknown'):
        """Check and correct tensor indices, ensuring they're within expected range."""
        tensor = self._ensure_int_indices(tensor, context)
        out_of_range = (tensor >= self.vocab_size) | (tensor < 0)
        if out_of_range.any():
            out_of_range_indices = tensor[out_of_range]
            raise ValueError(f"[{context}] Out of range indices found: {out_of_range_indices.unique().tolist()} in tensor with shape {tensor.shape}")
        return tensor

    def forward(self, x, mode='seq2seq', tgt=None):
        x = self._check_indices(x, context='input')
        if isinstance(mode, str):
            if mode == 'seq2seq':
                if tgt is None:
                    raise ValueError("tgt is None in seq2seq mode.")
                tgt = self._check_indices(tgt, context='target')
                if tgt is not None:
                    self._check_indices(tgt)
                    tgt = tgt.long()
                    x_embed = self.embedding(x)
                    tgt_embed = self.embedding(tgt)
                    memory = self.encoder(x_embed)
                    output = self.decoder(tgt_embed, memory)
                    output = self.seq2seq_output_layer(output)
                    return output
                else:
                    raise ValueError("No target provided for seq2seq mode.")
            else:
                x = self.embedding(x)
                x = self.base_model(x)
                sentiment = self.sentiment_analysis(x[:, 0, :])
                _, context = self.dialogue_contextualizer(x)
                entity_logits = self.entity_recognition_head(x)
                intent_logits = self.intent_classification_head(x[:, 0, :])
                entity_probs = F.softmax(entity_logits, dim=-1)
                intent_probs = F.softmax(intent_logits, dim=-1)
                return sentiment, context, entity_probs, intent_probs
        else:
            raise TypeError(f"Unexpected mode type: {type(mode)}")

    



#For later use, ignore this


#class TreeBasedRepresentation(nn.Module):
#    def __init__(self, num_node_types, embed_dim):
#        super(TreeBasedRepresentation, self).__init__()
#        self.node_embedding = nn.Embedding(num_node_types, embed_dim)
#        # Example for a simple Tree-LSTM setup
#        self.tree_lstm = nn.LSTMCell(embed_dim * 2, embed_dim)  # Assuming binary trees for simplicity
#
#    def forward(self, nodes, children):
#        """
#        nodes: Tensor of node indices [N]
#        children: List of tuples indicating the children indices for each node [N, 2]
#        """
#        embedded_nodes = self.node_embedding(nodes)
#        
#        # Example processing: iterate over nodes and process with Tree-LSTM
#        # Note: A real implementation would need to handle the tree structure,
#        # potentially recursively or with a stack/queue for breadth/depth-first processing
#        hx = torch.zeros(embedded_nodes.size(0), embedded_nodes.size(1))
#        cx = torch.zeros_like(hx)
#        
#        for i, node in enumerate(embedded_nodes):
#            child_indices = children[i]
#            child_cx = torch.sum(cx[child_indices], dim=0)
#            child_hx = torch.sum(hx[child_indices], dim=0)
#            hx[i], cx[i] = self.tree_lstm(torch.cat([node, child_hx]), (hx[i], child_cx))
#        
#        return hx
#
#
#class PythonModel(BaseModel):
#    def __init__(self, vocab_size, embed_dim, num_layers, heads, ff_dim, dropout_rate=0.1, num_node_types=None, code_vocab_size=None):
#        super(PythonModel, self).__init__(vocab_size, embed_dim, num_layers, heads, ff_dim, dropout_rate)
#        self.code_completion_head = nn.Linear(embed_dim, code_vocab_size or vocab_size)
#        self.error_detection_head = nn.Linear(embed_dim, 2)
#        
#        if num_node_types:
#            self.tree_based_representation = TreeBasedRepresentation(num_node_types, embed_dim)
#        
#        # Load advanced code embeddings if available
#        self.advanced_code_embeddings = nn.Embedding(code_vocab_size, embed_dim)
#        # Initialize weights of advanced_code_embeddings with pre-trained values if available
#
#    def forward(self, x, task_type='completion', tree=None, nodes=None, children=None):
#        # Assuming x is indices for advanced code embeddings
#        advanced_embeddings = self.advanced_code_embeddings(x)
#        
#        # Standard token embeddings + Advanced code embeddings
#        x = super().forward(advanced_embeddings)
#        
#        # Integrating syntax tree information if available
#        if tree is not None and hasattr(self, 'tree_based_representation') and nodes is not None and children is not None:
#            tree_repr = self.tree_based_representation(nodes, children)
#            x += tree_repr  # Combining tree representation with embeddings
#        
#        if task_type == 'completion':
#            logits = self.code_completion_head(x)
#            return F.log_softmax(logits, dim=-1)
#        elif task_type == 'error_detection':
#            logits = self.error_detection_head(x[:, 0, :])
#            return torch.sigmoid(logits)