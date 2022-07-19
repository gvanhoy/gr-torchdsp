from torch import nn
import torch

FFT_SIZE = 512


class FFT(nn.Module):
    def __init__(self):
        super(FFT, self).__init__()

    def forward(self, iq_data):
        torch.fft.fft(iq_data, dim=1, norm="ortho")
        return iq_data


x = torch.randn(1, FFT_SIZE, requires_grad=False,
                dtype=torch.cfloat)
model = FFT()
model.eval()

print(x.shape, model(x).shape)

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")
