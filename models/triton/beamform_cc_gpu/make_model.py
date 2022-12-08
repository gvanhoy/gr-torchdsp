from torch import nn
import numpy as np
import torch


class Beamform(nn.Module):
    def __init__(self):
        super(Beamform, self).__init__()

    def forward(self, iq_data, weights):
        out_real = torch.matmul(iq_data[:, :, ::2], weights[:, ::2])
        out_real = out_real - torch.matmul(iq_data[:, :, 1::2], weights[:, 1::2])

        out_imag = torch.matmul(iq_data[:, :, ::2], weights[:, 1::2])
        out_imag = out_imag + torch.matmul(iq_data[:, :, 1::2], weights[:, ::2])
        result = torch.cat([out_real, out_imag], dim=2)
        return result


x = torch.randn(1, 1000, 8, requires_grad=False, dtype=torch.float32)
weights = torch.randn(1, 8, 1, requires_grad=False, dtype=torch.float32)
model = Beamform()
model.eval()

model_output = model(x, weights)
print(x.shape, weights.shape, model_output.shape, model_output.flatten()[:10])


# Export the model
torch.onnx.export(model,               # model being run
                  (x, weights),                         # model input (or a tuple for multiple inputs)
                  "1/model.onnx",   # where to save the model (can be a file or file-like object)
                  verbose=True,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input__0', 'input__1'],   # the model's input names
                  output_names = ['output__0'], # the model's output names
                  dynamic_axes={'input__0' : {0 : 'batch_size'},
                                'input__1': {0: 'batch_size'},
                                'output__0' : {0 : 'batch_size'}}
                )

scripted = torch.jit.trace(model, (x, weights))
scripted.save("1/model.pt")

scripted_output = scripted(x, weights)
print(scripted_output.shape)
