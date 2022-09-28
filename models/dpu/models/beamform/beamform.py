from torch import nn
import numpy as np
import torch


class Beamform(nn.Module):
    """ 
        We implement narrowband beamforming of a complex signal
        using kernel size 8 and grouping 4 input channels
        to emulate a complex multiply (a + jb)*(c + jd) = (ac - bd) + j(bc + ad).

        Each 8 by 8 kernel will actually be redundant in dimension 1. So we expect the input will be 
        effectively "batched" along dimension 1 of the kernel (dimension 3 of the overall input).

        A single set of 4 real-only kernels that are 64 elements long are applied to 4 input 
        channels. Two of the input channels contain real components and two contain imaginary.
        Our calculation for throughput is essentially reduced by a factor of 4 because of this.

        We have 64 input channels to exploit input-channel parallelism and output-channel parallelism. 

        So, overall, we'll be "batching" input elements of 8 in three ways:
            1. In the input channel dimension, we batch by 16
            2. In the second kernel (width) dimension we batch by 8
            3. The actual data input here will batch in both height and width

    """
    def __init__(self, num_channels: int = 8):
        super(Beamform, self).__init__()
        batch_size = 1
        self.features = nn.Conv2d(
            num_channels*2*batch_size, 2*batch_size, kernel_size=1, stride=1, padding=0, groups=1, bias=False
        )

        # We effectively beamform across input channels for 8 inputs.
        # We assume 16 real-valued inputs representing 8 complex-valued inputs
        # and output 2 real-valued outputs representing 1 complex-valued output
        weights = np.ones((2*batch_size, num_channels*2*batch_size, 1, 1), dtype=np.float32)
        weights[1::2, ::2] = 0.0 # for every imaginary output, ignore the real inputs
        weights[::2, 1::2] = 0.0 # for every real output, ignore the imaginary inputs 
        
        self.features.weight = torch.nn.parameter.Parameter(
            torch.from_numpy(weights),
            requires_grad=False,
        )

    def forward(self, data):
        output = self.features(data)
        return output
