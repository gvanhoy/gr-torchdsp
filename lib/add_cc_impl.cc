/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "add_cc_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;

add_cc::sptr add_cc::make(unsigned int num_inputs, unsigned int device_num)
{
    return gnuradio::make_block_sptr<add_cc_impl>(num_inputs, device_num);
}

/*
 * The private constructor
 */
add_cc_impl::add_cc_impl(unsigned int num_inputs, unsigned int device_num)
    : gr::sync_block("add_cc",
                     gr::io_signature::make(num_inputs, num_inputs, sizeof(input_type)),
                     gr::io_signature::make(1, 1, sizeof(output_type))),
      d_num_inputs(num_inputs),
      d_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, device_num)
{
}

/*
 * Our virtual destructor.
 */
add_cc_impl::~add_cc_impl() {}

int add_cc_impl::work(int noutput_items, gr_vector_const_void_star& input_items, gr_vector_void_star& output_items)
{
    output_type* out = reinterpret_cast<output_type*>(output_items[0]);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

    std::vector<torch::Tensor> inputs;

    // We interpret all inputs as flattened Tensors of type Float32
    for (unsigned int input_idx = 0; input_idx < d_num_inputs; input_idx++) {
        input_type* in = reinterpret_cast<input_type*>(const_cast<void*>(input_items[input_idx]));
        auto input = torch::from_blob(reinterpret_cast<void*>(in), { noutput_items * 2 }, options);
        input.to(d_device);
        input.apply();
        inputs.push_back(input);
    }


    // Add all tensors together
    auto output = torch::zeros_like(inputs[0]);
    for (unsigned int input_idx = 0; input_idx < d_num_inputs; input_idx++) {
        output += inputs[input_idx];
    }

    // Copy the data back to CPU if it's not already there
    output.to(torch::kCPU);

    // Copy the raw Tensor data to the output.
    for (unsigned int idx = 0; idx < noutput_items; idx++) {
        out[idx].real(output[2 * idx].item().toFloat());
        out[idx].imag(output[2 * idx + 1].item().toFloat());
    }

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
