/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "fir_filter_ccc_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;

fir_filter_ccc::sptr fir_filter_ccc::make(const std::vector<gr_complex>& taps, unsigned int device_num)
{
    return gnuradio::make_block_sptr<fir_filter_ccc_impl>(taps, device_num);
}


/*
 * The private constructor
 */
fir_filter_ccc_impl::fir_filter_ccc_impl(const std::vector<gr_complex>& taps, unsigned int device_num)
    : gr::sync_block("fir_filter_ccc",
                     gr::io_signature::make(1, 1, sizeof(input_type)),
                     gr::io_signature::make(1, 1, sizeof(output_type))),
      d_device_num(device_num),
      d_device(::torch::cuda::is_available() ? ::torch::kCUDA : ::torch::kCPU, device_num)
{
    std::ostringstream msg;
    msg << "Running on device: " << d_device.type() << " Index: " << std::to_string(d_device.index()) << std::endl;
    GR_LOG_INFO(d_debug_logger, msg.str());

    auto tap_options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false).device(d_device);
    d_real_taps = torch::zeros({ static_cast<int>(taps.size()) }, tap_options);
    d_imag_taps = torch::zeros({ static_cast<int>(taps.size()) }, tap_options);

    for (int i = 0; i < taps.size(); i++) {
        d_real_taps.index_put_({ i }, taps[i].real());
        d_imag_taps.index_put_({ i }, taps[i].imag());
    }

    d_real_taps = d_real_taps.reshape({ 1, 1, -1 });
    d_imag_taps = d_real_taps.reshape({ 1, 1, -1 });

    set_history(taps.size());
}

/*
 * Our virtual destructor.
 */
fir_filter_ccc_impl::~fir_filter_ccc_impl() {}

int fir_filter_ccc_impl::work(int noutput_items,
                              gr_vector_const_void_star& input_items,
                              gr_vector_void_star& output_items)
{
    input_type* in = reinterpret_cast<input_type*>(const_cast<void*>(input_items[0]));
    output_type* out = reinterpret_cast<output_type*>(output_items[0]);

    auto options = torch::TensorOptions().dtype(::torch::kFloat32).requires_grad(false);
    auto conv_options = torch::nn::functional::Conv1dFuncOptions();

    // Interpret the in-buffer as a Tensor of type Float32 and transfer to the whatever device it needs to be on.
    auto input =
        ::torch::from_blob(reinterpret_cast<void*>(in), { (noutput_items + d_real_taps.size(2) - 1) * 2 }, options);
    input.to(d_device);

    auto real_input = input.index({ torch::indexing::Slice(0, torch::indexing::None, 2) }).reshape({ 1, 1, -1 });
    auto imag_input = input.index({ torch::indexing::Slice(1, torch::indexing::None, 2) }).reshape({ 1, 1, -1 });

    // We convolve with a stride of two with the real taps and the imag taps
    // We can probably reduce this to two convolutions by putting real/imag in channels
    // instead of entirely different tensors.
    auto conv_ac = torch::nn::functional::conv1d(real_input, d_real_taps, conv_options);
    auto conv_ad = torch::nn::functional::conv1d(real_input, d_imag_taps, conv_options);
    auto conv_bc = torch::nn::functional::conv1d(imag_input, d_real_taps, conv_options);
    auto conv_bd = torch::nn::functional::conv1d(imag_input, d_imag_taps, conv_options);

    // Then we need to add
    auto real_output = conv_ac - conv_bd;
    auto imag_output = conv_ad + conv_bc;

    // Transfer data back to CPU
    real_output.to(torch::kCPU);
    imag_output.to(torch::kCPU);

    // Flatten
    real_output = real_output.flatten();
    imag_output = imag_output.flatten();

    // This for-loop is because memcpy was being difficult without good reason.
    for (int i = 0; i < noutput_items; i++) {
        out[i].real(real_output.index({ i }).item().toFloat());
        out[i].imag(imag_output.index({ i }).item().toFloat());
    }

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
