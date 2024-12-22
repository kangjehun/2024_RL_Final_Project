import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .custom import Linear

class Transformer(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout
        self.num_layers = cfg.num_layers
        self.last_ln = cfg.last_ln
        
        self.pos_embs = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(self.dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(self.num_layers)]
        )
        
        if self.last_ln:
            self.last_norm_layer = nn.LayerNorm(self.d_model)
    
    def _generate_mask(self, T, H, W, device):
        
        N = H * W
        mask = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-1e10')).masked_fill(mask == 1, float(0.0)) # (T, T)
        # repeat the mask for spatial dimensions
        mask = torch.repeat_interleave(mask, N, dim=0) # (T, T) -> (T*N, T)
        mask = torch.repeat_interleave(mask, N, dim=1) # (T*N, T) -> (T*N, T*N)
        return mask
        
    def forward(self, z): # Delete Actions
        """
        B : Batch size
        T : Sequence length
        D : Model dimension
        H : Height
        W : Width
        L : Number of layers
        T': T*H*W 
        """
        
        B, T, D, H, W = z.shape
        
        # generate mask
        attn_mask = self._generate_mask(T, H, W, z.device) # (T*N, T*N) where N = H*W
        
        # create positional embeddings
        pos_inputs = torch.arange(T*H*W, dtype=torch.float).to(z.device)
        pos_embs = self.drop(self.pos_embs(pos_inputs)) # TODO [CHECK] Why dropout?
        
        # flatten spatial dimensions
        z = rearrange(z, 'b t d h w -> (t h w) b d')
        input = z + pos_embs
        
        # T, B, D(d_model)
        output = input
        output_list = []
        for i, layer in enumerate(self.layers):
            output = layer(output, attn_mask=attn_mask) # T', B, D
            output_list.append(output)
        output = torch.stack(output_list, dim=1) # T', L, B, D
        output = rearrange(output, '(t h w) l b d -> b t l d h w', h=H, w=W)
        
        return output # B, T, L, D, H, W
           
class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model):
        
        super().__init__()
        
        self.d_model = d_model
        inv_freq = 1/ (10000 ** (torch.arange(0.0, d_model, 2.0) / self.d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, inputs):
        
        # TODO Positional embedding order? [sin, sin, ... , cos, cos, ...] -> [sin, cos, sin, cos, ...]
        sinusoid_input = torch.einsum('i , j -> i j', inputs, self.inv_freq)
        pos_emb = torch.cat([sinusoid_input.sin(), sinusoid_input.cos()], dim=-1)
        return pos_emb[:, None, :]
    
class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        self.mha = MultiheadAttention(cfg)
        self.pos_ff = PositionwiseFeedForward(cfg)
        
    def forward(self, inputs, attn_mask=None):
        
        # Self-attention
        # TODO [REMINDER] I removed GRUGating mechanism
        residual = inputs
        x = self.mha(inputs, inputs, inputs, attn_mask=attn_mask)
        x = x + residual
        residual = x
        x = self.pos_ff(x)
        x = x + residual
        return x
        

class MultiheadAttention(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head
        self.num_heads = cfg.num_heads
        self.weight_init = cfg.weight_init
        self.dropout = cfg.dropout
        self.dropatt = cfg.dropatt
        self.pre_lnorm = cfg.pre_lnorm
        
        self.q_net = Linear(self.d_model, self.d_head * self.num_heads, bias=False, weight_init=self.weight_init)
        self.k_net = Linear(self.d_model, self.d_head * self.num_heads, bias=False, weight_init=self.weight_init)
        self.v_net = Linear(self.d_model, self.d_head * self.num_heads, bias=False, weight_init=self.weight_init)
        self.out_net = Linear(self.d_head * self.num_heads, self.d_model, bias=False, weight_init=self.weight_init)
        
        self.drop = nn.Dropout(self.dropout)
        self.dropatt = nn.Dropout(self.dropatt)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.scale = 1 / (self.d_head ** 0.5)
    
    def forward(self, query, key, value, attn_mask=None):
        """
        T = t*h*w   : Length of Query, Key, Value (T_q, T_k, T_v) 
        B           : Batch size
        d_model     : transformer model dimension
        
        query, key, value : (T, B, d_model)
        attn_mask         : (T_q, T_k)
        """
        
        T_q, bsz = query.shape[:2]
        T_k, bsz = key.shape[:2]
        
        # Project the queries, keys, and values
        if self.pre_lnorm:
            q = self.q_net(self.layer_norm(query))
            k = self.k_net(self.layer_norm(key))
            v = self.v_net(self.layer_norm(value))
        else:
            q = self.q_net(query)
            k = self.k_net(key)
            v = self.v_net(value)
        
        # Reshape the queries, keys, and values
        q = q.view(T_q, bsz, self.num_heads, self.d_head) # (T_q, B, num_heads, d_head)
        k = k.view(T_k, bsz, self.num_heads, self.d_head) # (T_k, B, num_heads, d_head)
        v = v.view(T_k, bsz, self.num_heads, self.d_head) # (T_k, B, num_heads, d_head), T_k = T_v
        
        # attn_score = QK^T/sqrt(d)
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (q, k)) * self.scale
        
        # apply mask
        if attn_mask is not None:
            attn_score = attn_score.float().masked_fill(
                attn_mask[:, :, None, None].bool(), -float('inf')).type_as(attn_score)
        
        # softmax(QK^T/sqrt(d)) with dropout
        attn_prob = F.softmax(attn_score, dim=1) # [T_q, "T_k", B, num_heads]
        attn_prob = self.dropatt(attn_prob)
        
        # softmax(QK^T/sqrt(d)) * V
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, v))
        
        # Concat(head_1, head_2, ..., head_h)
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.num_heads * self.d_head)
        
        # Multihead(Q, K, V) = Concat(head1, head2, ..., head_h) * W0
        attn_out = self.out_net(attn_vec)
        attn_out = self.drop(attn_out)
        
        if self.pre_lnorm:
            output = attn_out
        else:
            output = self.layer_norm(attn_out)
        
        return output 

class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        self.d_model = cfg.d_model
        self.d_ff  = cfg.d_ff
        self.dropout = cfg.dropout
        self.pre_lnorm = cfg.pre_lnorm
        
        self.net = nn.Sequential(
            Linear(self.d_model, self.d_ff),
            nn.ReLU(inplace=True),
            Linear(self.d_ff, self.d_model),
            nn.Dropout(self.dropout)
        )
        
        self.layer_norm = nn.LayerNorm(self.d_model)
    
    def forward(self, input):
        
        if self.pre_lnorm:
            output = self.net(self.layer_norm(input))
        else:
            output = self.layer_norm(self.net(input))
        
        return output