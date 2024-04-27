import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, vocab_size, max_len, n_segments, dim, p_drop_hidden=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim)  # token embedding
        self.pos_embed = nn.Embedding(max_len, dim)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, dim)  # segment(token type) embedding

        self.norm = LayerNorm(dim)
        self.drop = nn.Dropout(p_drop_hidden)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        e = self.tok_embed(x) + Variable(self.pe[:, :x.size(1)], requires_grad=False) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.scores = None  # for visualization
        self.n_heads = n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :].float()
            else:
                mask = mask[:, None, :, :].float()
            # scores = scores.masked_fill(mask==0, -1e9)
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

class MultiHeadedCrossAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.scores = None  # for visualization
        self.n_heads = n_heads

    def forward(self, x_q, x_k, x_v, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x_q), self.proj_k(x_k), self.proj_v(x_v)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :].float()
            else:
                mask = mask[:, None, :, :].float()
            # scores = scores.masked_fill(mask==0, -1e9)
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, dim, dim_ff):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, dim, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, n_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = LayerNorm(dim)
        self.pwff = PositionWiseFeedForward(dim, dim_ff)
        self.norm2 = LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class DecBlock(nn.Module):
    """ Transformer Decoder Block """
    def __init__(self, dim, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, n_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = LayerNorm(dim)
        self.pwff = PositionWiseFeedForward(dim, dim_ff)
        self.norm2 = LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        # Cross Attention Sub Module
        self.cross_attn = MultiHeadedCrossAttention(dim, n_heads, dropout)
        self.proj_2 = nn.Linear(dim, dim)
        self.norm3 = LayerNorm(dim)

    def forward(self, enc_out, src_mask, trg, trg_mask):
        trg_1 = self.attn(trg, trg_mask)
        trg_2 = self.norm1(trg + self.drop(self.proj(trg_1)))
        trg_3 = self.cross_attn(trg_2, enc_out, enc_out, src_mask)
        trg_4 = self.norm3(trg_2 + self.drop(self.proj_2(trg_3)))
        trg_5 = self.norm2(trg_4 + self.drop(self.pwff(trg_4)))
        return trg_5


class Decoder(nn.Module):
    """Transformer Decoder with Cross-Attention Blocks"""

    def __init__(self, dim, n_heads, dim_ff, blocks, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([DecBlock(dim, n_heads, dim_ff, dropout) for _ in range(blocks)])
        self.norm = LayerNorm(dim)

    def forward(self, enc_out, src_mask, trg, trg_mask):
        trg = self.norm(trg)
        for block in self.blocks:
            trg = block(enc_out, src_mask, trg, trg_mask)
        return trg


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, dim, n_heads, dim_ff, blocks, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, n_heads, dim_ff, dropout) for _ in range(blocks)])
        self.norm = LayerNorm(dim)

    def forward(self, h, mask):
        h = self.norm(h)
        for block in self.blocks:
            h = block(h, mask)
        return h

