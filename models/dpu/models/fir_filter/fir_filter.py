from torch import nn
import numpy as np
import torch


class FIRFilter(nn.Module):
    """ 

    """
    def __init__(self, taps: np.ndarray):
        super(FIRFilter, self).__init__()
        taps = np.pad(taps, (0, 64*16 - taps.shape[0]))

        self.conv = nn.Conv2d(
            256, 16, kernel_size=8, stride=8, padding=0, groups=16, bias=False
        )

        weights = taps[:64]
        weights = weights.reshape(1, 1, 8, 8)
        for idx in range(1, 16):
            tap_section = taps[64*idx:64*(idx + 1)]
            tap_section = tap_section.reshape(1, 1, 8, 8)
            weights = np.concatenate((weights, tap_section), axis=0)

        weights = np.repeat(weights, repeats=16, axis=1)
        self.conv.weight = torch.nn.parameter.Parameter(
            torch.from_numpy(weights.astype(np.float32)),
            requires_grad=False,
        )

    def forward(self, data):
        return self.conv(data)