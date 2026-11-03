import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from src.positional_encodings import PEEncoder

L_MAX = 100              # clamp for position offsets in the attention bias


class MultiEmb(nn.Module):
    def __init__(self, sizes, dim):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, dim) for s in sizes])

    def forward(self, x):
        return sum(e(x[:, i]) for i, e in enumerate(self.embs))


class GPSLayer(nn.Module):
    def __init__(self, dim, heads, dropout, attn_dropout, use_bias):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.local = GINEConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)),
                              train_eps=True)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.norm_l, self.norm_a, self.norm_f = (nn.BatchNorm1d(dim) for _ in range(3))
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(2 * dim, dim), nn.Dropout(dropout))
        self.T = nn.Parameter(torch.zeros(2 * L_MAX + 1)) if use_bias else None

    def forward(self, h, edge_index, e, batch, same=None, off=None):
        hl = self.local(h, edge_index, e)
        hl = self.norm_l(F.dropout(hl, self.dropout, self.training) + h)

        hd, mask = to_dense_batch(h, batch)
        kpm = torch.zeros(mask.shape, device=h.device).masked_fill(~mask, float("-inf"))
        am = None
        if self.T is not None:
            bias = self.T[off] * same                       # (B, N, N)
            am = bias.repeat_interleave(self.heads, dim=0)   # (B*heads, N, N)
        ha, _ = self.attn(hd, hd, hd, attn_mask=am, key_padding_mask=kpm, need_weights=False)
        ha = self.norm_a(F.dropout(ha[mask], self.dropout, self.training) + h)

        out = hl + ha
        return self.norm_f(out + self.ff(out))


class Net(nn.Module):
    def __init__(self, node_sizes, edge_sizes, out_dim, pe, bias, dim, layers, heads,
                 dropout, attn_dropout):
        super().__init__()
        self.atom, self.bond = MultiEmb(node_sizes, dim), MultiEmb(edge_sizes, dim)
        self.pe_enc = PEEncoder(pe, dim) if pe != "none" else None
        self.use_bias = bias
        self.layers = nn.ModuleList([GPSLayer(dim, heads, dropout, attn_dropout, bias)
                                     for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, out_dim))

    def forward(self, data):
        h = self.atom(data.x)
        if self.pe_enc is not None:
            h = h + self.pe_enc(data)
        e = self.bond(data.edge_attr)
        same = off = None
        if self.use_bias:
            id_d, mask = to_dense_batch(data.pcid, data.batch, fill_value=-1)
            pos_d, _ = to_dense_batch(data.pcpos, data.batch, fill_value=0)
            same = (id_d[:, :, None] == id_d[:, None, :]) & mask[:, :, None] & mask[:, None, :]
            off = (pos_d[:, :, None] - pos_d[:, None, :]).clamp(-L_MAX, L_MAX) + L_MAX
        for layer in self.layers:
            h = layer(h, data.edge_index, e, data.batch, same, off)
        return self.head(global_mean_pool(h, data.batch))
