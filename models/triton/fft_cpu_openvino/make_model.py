from torch import nn
import torch

FFT_SIZE = 512


class FFT(nn.Module):
    def __init__(self):
        super(FFT, self).__init__()

    def forward(self, iq_data):
        result = torch.roll(torch.fft.fft(
            iq_data, dim=2, norm="ortho"),
            FFT_SIZE // 2
        )
        # We do this because TIS doesn't like complex outputs sometimes
        result = torch.cat([result.real, result.imag], dim=1)
        return result.permute((0, 2, 1))


x = torch.randn(1, 1, FFT_SIZE, requires_grad=False,
                dtype=torch.cfloat)


model = FFT()
model.eval()

print(x.shape, model(x).shape)

scripted = torch.jit.trace(model, [x])
scripted.save("1/model.pt")
