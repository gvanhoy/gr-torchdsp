from torch import nn
import torch


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.add = torch.randn(1, 16, requires_grad=False)

    def forward(self, x): 
        return x + self.add


x = torch.randn(1, 16, requires_grad=False)
model = Add()
model.eval()

print(x.shape, model(x).shape)

scripted = torch.jit.trace(model, x)
scripted.save("1/model.pt")

