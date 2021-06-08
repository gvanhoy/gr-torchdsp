/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "multiply_const_vcvc_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;

multiply_const_vcvc::sptr multiply_const_vcvc::make(const std::vector<gr_complex>& constant, unsigned int device_num)
{
    return gnuradio::make_block_sptr<multiply_const_vcvc_impl>(constant, device_num);
}


/*
 * The private constructor
 */
multiply_const_vcvc_impl::multiply_const_vcvc_impl(const std::vector<gr_complex>& constant, unsigned int device_num)
    : gr::sync_block("multiply_const_vcvc",
                     gr::io_signature::make(1, 1, sizeof(input_type) * constant.size()),
                     gr::io_signature::make(1, 1, sizeof(output_type) * constant.size())),
      d_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, device_num)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    d_constant_real = torch::zeros({ static_cast<int>(constant.size()) }, options);
    d_constant_imag = torch::zeros({ static_cast<int>(constant.size()) }, options);

    for (int idx = 0; idx < constant.size(); idx++) {
        d_constant_real.index_put_({ idx }, constant[idx].real());
        d_constant_imag.index_put_({ idx }, constant[idx].imag());
    }

    d_constant_real.to(d_device);
    d_constant_imag.to(d_device);
}

/*
 * Our virtual destructor.
 */
multiply_const_vcvc_impl::~multiply_const_vcvc_impl() {}

int multiply_const_vcvc_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    input_type* in = reinterpret_cast<input_type*>(const_cast<void*>(input_items[0]));
    output_type* out = reinterpret_cast<output_type*>(output_items[0]);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

    for (unsigned int idx = 0; idx < noutput_items; idx++) {
        // We interpret all inputs as flattened Tensors of type Float32
        auto input = torch::from_blob(reinterpret_cast<void*>(in), { d_constant_imag.size(0) * 2 }, options);
        input.to(d_device);

        // Multiply tensors element-wise
        auto real_input = input.index({ torch::indexing::Slice(0, torch::indexing::None, 2) });
        auto imag_input = input.index({ torch::indexing::Slice(1, torch::indexing::None, 2) });
        auto output_real = real_input * d_constant_real - imag_input * d_constant_imag;
        auto output_imag = real_input * d_constant_imag + imag_input * d_constant_real;

        // Copy the data back to CPU if it's not already there
        output_real.to(torch::kCPU);
        output_imag.to(torch::kCPU);

        // Copy the raw Tensor data to the output.
        for (unsigned int idx = 0; idx < d_constant_imag.size(0); idx++) {
            out[idx].real(output_real[idx].item().toFloat());
            out[idx].imag(output_imag[idx].item().toFloat());
        }

        // Increment pointers
        in += d_constant_imag.size(0);
        out += d_constant_imag.size(0);
    }

    return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
