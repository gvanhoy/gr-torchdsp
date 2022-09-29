from scipy.linalg import dft
from torch import nn
import numpy as np
import torch


class DFT(nn.Module):
    """

    So what we do here, is distribute DFT coefficients across dft_size channels. 
    Ultimately we end up with the same number of output channels each of which are a linear combination
    of one of the rows in a DFT matrix.

    We batch in the height and width dimensions of the input,
    
    To do a complex DFT, we have to use groups=4 and there needs to be 
    a redundancy of 2 in the input.

    (in_re + j*in_im)(dft_re + j*dft_im) = 
        out_real = in_re*dft_re - in_im*dft_im
        out_imag = in_re*dft_im + in_im*dft_re

    ac = weights[::2, ::2]
    db = weights[1::2, ::2]

        weights[out_channels, in_channels]
    """
    def __init__(self, dft_size: int = 8):
        super(DFT, self).__init__()
        dft_matrix = dft(dft_size)

        self.conv = nn.Conv2d(
            2*dft_size, 2*dft_size, kernel_size=1, stride=1, padding=0, groups=1, bias=False
        )
        weights_real = dft_matrix.real.reshape(dft_size, dft_size, 1, 1).astype(np.float32)
        weights_imag = dft_matrix.imag.reshape(dft_size, dft_size, 1, 1).astype(np.float32)

        weights = np.zeros((2*dft_size, 2*dft_size, 1, 1), dtype=np.float32)
        
        # Weights for real output
        weights[::2, ::2] = weights_real
        weights[::2, 1::2] = -weights_imag

        # Weights for imag output
        weights[1::2, ::2] = weights_imag
        weights[1::2, 1::2] = weights_real

        self.conv.weight = torch.nn.parameter.Parameter(
            torch.from_numpy(weights.astype(np.float32)),
            requires_grad=False,
        )

    def forward(self, data):
        return self.conv(data)
