from torch import nn
import torch


class AddConst(nn.Module):
    def __init__(self):
        super(AddConst, self).__init__()

    def forward(self, x, y):
        return x + y


x = torch.randn(1, 1000, requires_grad=False)
y = torch.randn(1, 1000, requires_grad=False)
model = AddConst()
model.eval()

print(x.shape, model(x, y).shape)

scripted = torch.jit.trace(model, (x, y))
scripted.save("1/model.pt")
