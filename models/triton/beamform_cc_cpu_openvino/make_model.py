from array import array
from torch import nn
import torch


class Beamform(nn.Module):
    def __init__(self):
        super(Beamform, self).__init__()

    def forward(
        self,
        in0_real,
        in0_imag,
        in1_real,
        in1_imag,
        in2_real,
        in2_imag,
        in3_real,
        in3_imag,
        bf_real,
        bf_imag
    ):
        in_real = torch.cat((in0_real, in1_real, in2_real, in3_real), dim=1)
        in_imag = torch.cat((in0_imag, in1_imag, in2_imag, in3_imag), dim=1)

        a = torch.matmul(bf_real, in_real)
        d = torch.matmul(bf_imag, in_imag)

        b = torch.matmul(bf_imag, in_real)
        c = torch.matmul(bf_real, in_imag)
        out = torch.cat((torch.mul(a, c) - torch.mul(b, d),
                        torch.mul(a, d) + torch.mul(b, c)), dim=1)
        out = out.permute((0, 2, 1))
        return out


x = torch.randn(1, 1, 1000, requires_grad=False,
                dtype=torch.float32)

bf = torch.randn(1, 1, 4, requires_grad=False,
                 dtype=torch.float32)

model = Beamform()
model.eval()

print(x.shape, model(x, x, x, x, x, x, x, x, bf, bf)[0].shape)

scripted = torch.jit.trace(
    model, (x, x, x, x, x, x, x, x, bf, bf))
scripted.save("1/model.pt")
