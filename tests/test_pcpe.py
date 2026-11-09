import torch
from types import SimpleNamespace

from src.positional_encodings import MAX_TRACKS, PEEncoder, sinusoidal


def test_sinusoidal_shape():
    assert sinusoidal(torch.arange(5), 8).shape == (5, 8)


def test_pcpe_encoder_output_shape_and_clamping():
    enc = PEEncoder("pcpe", 16)
    data = SimpleNamespace(
        pcid=torch.tensor([0, 1, 500]),
        pcpos=torch.tensor([0, 1, 2])
    )
    assert enc(data).shape == (3, 16)
    assert MAX_TRACKS == 64
