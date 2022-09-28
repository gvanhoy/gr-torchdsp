from torch import nn
import torch


class MatrixMultiply(nn.Module):
    def __init__(self):
        super(MatrixMultiply, self).__init__()

    def forward(self, x, y):
        return x*y


x = torch.randn(1, 4, 16, requires_grad=False,
                dtype=torch.float32)
model = MatrixMultiply()
model.eval()

print(x.shape, model(x, x).shape)

scripted = torch.jit.trace(model, [x, x])
scripted.save("1/model.pt")
