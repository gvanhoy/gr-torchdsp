
from torch_rfml_train import VTCNNModule
import pytorch_lightning as pl
import torch

# These checkpoints can be found in the lightning_logs/ folder
# created when training the model
module = VTCNNModule.load_from_checkpoint("vt_cnn2.ckpt")

# The trace method here freezes the batch size of the
# saved model.
batch_size = 128
inputs = torch.randn((batch_size, 2, 128))
script = module.to_torchscript(method="trace", example_inputs=inputs)
torch.jit.save(script, "vt_cnn2.pt")
