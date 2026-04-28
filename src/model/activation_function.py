"""Some custom activation functions"""

import torch
from torch import nn


class ComplexGELU(nn.Module):
    """Complex GELU"""

    def __init__(self):
        super().__init__()

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Just applies GELU to real and imagine parts"""
        if not x.is_complex():
            raise NotImplementedError("Not implemented for non-complex tensors")
        return torch.complex(self.gelu(x.real), self.gelu(x.imag))
