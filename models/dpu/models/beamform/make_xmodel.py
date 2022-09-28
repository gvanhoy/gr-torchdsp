from pytorch_nndct.apis import torch_quantizer, Inspector
from beamform import Beamform
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import subprocess
import torch
import tqdm


def inspect(
    model: nn.Module,
    batch_size: int,
    element_shape: list,
    device: torch.DeviceObjType,
    arch_identifer: str,
):
    shape = (batch_size, *element_shape)
    dummy_data = torch.randn(shape, device=device)

    inspector = Inspector(arch_identifer)
    inspector.inspect(model, (dummy_data,), device=torch.device("cuda:0"))


def quantize(
    model: nn.Module,
    batch_size: int,
    element_shape: list,
    device: torch.DeviceObjType,
    calibrate: bool = True,
):
    shape = (batch_size, *element_shape)
    dummy_data = torch.randn(shape, device=device)

    quantizer = torch_quantizer(
        "calib" if calibrate else "test", model, (dummy_data,), device=device
    )

    quant_model = quantizer.quant_model

    # Yes, you need to do this in calibration AND deploy/test mode.
    for i in tqdm.tqdm(range(1000), desc="calibrating" if calibrate else "exporting"):
        dummy_data = torch.randn((shape), device=device)
        dummy_data.to(device=device)
        output = quant_model.forward(dummy_data)

    if calibrate:
        quantizer.export_quant_config()
        return

    compare(model, quant_model)
    quantizer.export_xmodel(deploy_check=True)


def compare(model: nn.Module, quant: nn.Module):
    # I guess we're doing real-only beamforming, because we just sum.
    num_channels = 8
    width_batch_size = 128
    height_batch_size = 128
    batch_size = 1
    normal_data = np.exp(
        2j*np.arange(batch_size * width_batch_size * height_batch_size)/255)

    # Real
    normal_data_real = normal_data.real
    normal_data_real = normal_data_real.reshape(
        -1, height_batch_size, width_batch_size)

    # Imag
    normal_data_imag = normal_data.imag
    normal_data_imag = normal_data_imag.reshape(
        -1, height_batch_size, width_batch_size)

    input_data = np.concatenate([
        normal_data_real,
        normal_data_imag,
    ])
    input_data = np.tile(input_data, reps=(num_channels, 1, 1))
    print(input_data.shape)

    input_data = torch.from_numpy(
        input_data.reshape(1, 2*num_channels, height_batch_size,
                           width_batch_size).astype(np.float32)
    ).cuda()

    # quant_output = model(input_data)
    quant_output = quant((2**-1)*input_data)
    print(quant_output.shape)

    plt.figure(figsize=(16, 9))
    quant_output = quant_output.detach().cpu().numpy()
    for idx in range(quant_output.shape[1]):
        # for idx in range(1):
        plt.plot(quant_output[:, idx].reshape(-1), marker="*")

    plt.savefig("test.png", dpi=160)


def deploy(architecture: str):
    command = (
        "vai_c_xir -x quantize_result/{} -n {} -o {} -a ../../architectures/{}".format(
            "Beamform_int.xmodel", "beamform", "quantize_result", architecture
        )
    )
    print(command)
    subprocess.call(command, shell=True)


def main():
    num_channels = 8
    model = Beamform(num_channels=num_channels)
    model.to(device=torch.device("cuda:0"))
    model.eval()

    # Not working for depth-wise convolution...
    # inspect(
    #     model,
    #     batch_size=1,
    #     element_shape=(16, 8 * 16, 8 * 16),
    #     device=torch.device("cuda:0"),
    #     arch_identifer="DPUCZDX8G_ISA1_B4096",
    # )
    quantize(
        model,
        batch_size=16,
        element_shape=(num_channels * 2, num_channels * 16, num_channels * 16),
        device=torch.device("cuda:0"),
        calibrate=True,
    )
    quantize(
        model,
        batch_size=1,
        element_shape=(num_channels * 2, num_channels * 16, num_channels * 16),
        device=torch.device("cuda:0"),
        calibrate=False,
    )
    deploy("kv260_arch.json")


if __name__ == "__main__":
    main()
