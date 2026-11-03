import math

import torch
import torch.nn as nn

K_LAP, K_RW = 8, 16
MAX_TRACKS = 64          # track-id embedding table size (ids >= 63 are clamped)


def sinusoidal(pos, dim):
    i = torch.arange(dim // 2, device=pos.device).float()
    freq = torch.exp(-math.log(10000.0) * (2 * i / dim))
    ang = pos.float()[:, None] * freq[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class PEEncoder(nn.Module):
    def __init__(self, pe, hidden, id_dim=8, pos_dim=8):
        super().__init__()
        self.pe = pe
        if pe in ("lap", "rwse"):
            in_dim = K_LAP if pe == "lap" else K_RW
        else:
            in_dim = id_dim + pos_dim
            self.id_emb = nn.Embedding(MAX_TRACKS, id_dim)
            self.pos_dim = pos_dim
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, data):
        if self.pe in ("lap", "rwse"):
            pe = data.pe
            if self.pe == "lap" and self.training:
                s = (torch.randint(0, 2, (data.num_graphs, pe.size(1)), device=pe.device) * 2 - 1).float()
                pe = pe * s[data.batch]
        else:
            ids = data.pcid.clamp(max=MAX_TRACKS - 1)
            pe = torch.cat([self.id_emb(ids), sinusoidal(data.pcpos, self.pos_dim)], dim=-1)
        return self.proj(pe)
