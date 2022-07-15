from torch import nn
import torch


class FFT(nn.Module):
    def __init__(self):
        super(FFT, self).__init__()

    def forward(self, x_real, x_imag):
        raw_iq = x_real + 1j*x_imag
        fft = torch.fft.fft(raw_iq, dim=1, norm="ortho")
        return torch.cat([fft.real, fft.imag], dim=1)


batch_size = 1

x = torch.randn(batch_size, 512, requires_grad=False)
model = FFT()
model.eval()

print(x.shape, model(x, x).shape)

scripted = torch.jit.trace(model, [x, x] )
scripted.save("1/model.pt")

