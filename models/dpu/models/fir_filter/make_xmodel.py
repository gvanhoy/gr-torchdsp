from pytorch_nndct.apis import torch_quantizer, Inspector
from fir_filter import FIRFilter
from torch import nn
from scipy import signal as sp
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

    quantizer.export_xmodel(deploy_check=True)


def deploy(architecture: str):
    command = (
        "vai_c_xir -x quantize_result/{} -n {} -o {} -a ../../architectures/{}".format(
            "FIRFilter_int.xmodel", "fir_filter", "quantize_result", architecture
        )
    )
    print(command)
    subprocess.call(command, shell=True)


def main():
    low_pass_filter = sp.firwin(64*16 - 1, 1/64.0, 1/128.0)
    model = FIRFilter(taps=low_pass_filter)
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
        element_shape=(256, 8 * 16, 8 * 16),
        device=torch.device("cuda:0"),
        calibrate=True,
    )
    quantize(
        model,
        batch_size=1,
        element_shape=(256, 8 * 16, 8 * 16),
        device=torch.device("cuda:0"),
        calibrate=False,
    )
    deploy("kv260_arch.json")


if __name__ == "__main__":
    main()
