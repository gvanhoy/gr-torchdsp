from torch import nn
import numpy as np
import torch


class Beamform(nn.Module):
    def __init__(self):
        super(Beamform, self).__init__()
        self.bf = torch.ones((1, 8), dtype=torch.float32)
        # self.bf[1::2] = 0.0

    def forward(
        self,
        in0,
        in1,
        in2,
        in3
    ):
        in_real = torch.cat((in0[::2], in1[::2], in2[::2],
                            in3[::2]), dim=0).reshape(-1, 4, 1000)
        in_imag = torch.cat((in0[1::2], in1[1::2], in2[1::2],
                            in3[1::2]), dim=0).reshape(-1, 4, 1000)

        bf_real = self.bf[0, ::2].reshape(-1, 1, 4)
        bf_imag = self.bf[0, 1::2].reshape(-1, 1, 4)
        out_real = torch.matmul(bf_real, in_real) - \
            torch.matmul(bf_imag, in_imag)
        out_imag = torch.matmul(bf_imag, in_real) + \
            torch.matmul(bf_real, in_imag)
        result = torch.cat([out_real, out_imag], dim=0)

        # in_matrix = torch.cat((in0, in1, in2, in3), dim=0).reshape(-1, 4, 1000)
        # out = torch.matmul(self.bf, in_matrix)
        # result = torch.cat([out.real, out.imag], dim=0)
        return result


x = torch.randn(4, 2000, requires_grad=False,
                dtype=torch.float32)

model = Beamform()
model.eval()

model_output = model(x[0], x[1],
                     x[2], x[3])
print(x.shape, model_output.shape, model_output.flatten()[:10])

scripted = torch.jit.trace(model, (x[0], x[1],
                                   x[2], x[3]))
scripted.save("1/model.pt")

scripted_output = scripted(x[0], x[1],
                           x[2], x[3])
print(scripted_output.shape)
