from torch import nn
from scipy import signal as sp
import numpy as np
import torch


class FIRFilter(nn.Module):
    def __init__(self, taps: torch.tensor):
        super(FIRFilter, self).__init__()
        self.taps = taps

    def forward(self, iq_data):
        return nn.functional.conv1d(iq_data, self.taps)


x = torch.randn(1, 1, 1024, requires_grad=False,
                dtype=torch.float32)

taps = sp.firwin(64, 1/8., 1/16.0, window="kaiser").astype(np.float32)
torch_taps = torch.from_numpy(taps)
torch_taps.requires_grad = False
model = FIRFilter(torch_taps.reshape(1, 1, -1))

model.eval()

print(x.shape, model(x).shape)

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")
