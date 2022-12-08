from pytorch_nndct.apis import torch_quantizer, Inspector
from matplotlib import pyplot as plt
from dft import DFT
from torch import nn
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
    # dummy_data = torch.ones(shape, device=device)
    # dummy_data = torch.randint(-127, 128, size=shape, device=device)

    quantizer = torch_quantizer(
        "calib" if calibrate else "test", model, (dummy_data, ), device=device
    )

    quant_model = quantizer.quant_model

    # Yes, you need to do this in calibration AND deploy/test mode.
    for i in tqdm.tqdm(range(1000), desc="calibrating" if calibrate else "exporting"):
        dummy_data = torch.randn((shape), device=device)
        dummy_data.to(device=device)
        output = quant_model.forward(dummy_data,)
    print(
        dummy_data.shape,
        quant_model.forward(torch.randn(shape, device=device)).shape
    )
    # print(
    #     model.forward(torch.randn(shape, device=device))[0],
    #     quant_model.forward(torch.randn(shape, device=device))[0]
    # )

    if calibrate:
        quantizer.export_quant_config()
        return

    compare(model, quant_model)
    quantizer.export_xmodel(deploy_check=True)

def compare(model: nn.Module, quant: nn.Module):
    # I guess we're doing real-only beamforming, because we just sum.
    dft_size = 64
    width_batch_size = 1
    height_batch_size = 1
    normal_data = np.exp(2j*np.arange(dft_size * width_batch_size * height_batch_size)/8)

    # Real
    normal_data_real = normal_data.real
    normal_data_real = normal_data_real.reshape(dft_size, height_batch_size, width_batch_size)

    # Imag
    normal_data_imag = normal_data.imag
    normal_data_imag = normal_data_imag.reshape(dft_size, height_batch_size, width_batch_size)
    
    input_data = np.zeros((2*dft_size, height_batch_size, width_batch_size), dtype=np.float32)
    input_data[::2] = normal_data_real
    input_data[1::2] = normal_data_imag

    input_data = torch.from_numpy(
        input_data.reshape(1, 2*dft_size, height_batch_size, width_batch_size).astype(np.float32)
    ).cuda()

    quant_output = model(input_data)
    # quant_output = quant((2**-1)*input_data)
    print(quant_output.shape)

    plt.figure(figsize=(16, 9))
    quant_output = quant_output.detach().cpu().numpy()
    real_output = quant_output[:, ::2].reshape(-1)
    imag_output = quant_output[:, 1::2].reshape(-1)
    # plt.plot(real_output, marker="*")
    # plt.plot(imag_output, marker="*")
    plt.plot(10*np.log10(np.abs(real_output + 1j*imag_output)))
    # _ = plt.psd(normal_data)

    plt.savefig("test.png", dpi=160)

def deploy(architecture: str):
    command = (
        "vai_c_xir -x quantize_result/{} -n {} -o {} -a ../../architectures/{}".format(
            "DFT_int.xmodel", "dft", "quantize_result", architecture
        )
    )
    print(command)
    subprocess.call(command, shell=True)


def main():
    dft_size = 64
    model = DFT(dft_size=dft_size)
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
        element_shape=(dft_size * 2, 16, 16),
        device=torch.device("cuda:0"),
        calibrate=True,
    )
    quantize(
        model,
        batch_size=1,
        element_shape=(dft_size * 2, 16, 16),
        device=torch.device("cuda:0"),
        calibrate=False,
    )
    deploy("u50_arch.json")


if __name__ == "__main__":
    main()
