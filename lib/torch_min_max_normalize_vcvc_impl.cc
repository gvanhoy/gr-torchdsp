/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "torch_min_max_normalize_vcvc_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;

torch_min_max_normalize_vcvc::sptr
torch_min_max_normalize_vcvc::make(unsigned int device_num) {
  return gnuradio::make_block_sptr<torch_min_max_normalize_vcvc_impl>(
      device_num);
}

/*
 * The private constructor
 */
torch_min_max_normalize_vcvc_impl::torch_min_max_normalize_vcvc_impl(
    unsigned int device_num)
    : gr::sync_block("torch_min_max_normalize_vcvc",
                     gr::io_signature::make(1, 1, sizeof(input_type)),
                     gr::io_signature::make(1, 1, sizeof(output_type))),
      d_device_num(device_num),
      d_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU,
               device_num) {}

/*
 * Our virtual destructor.
 */
torch_min_max_normalize_vcvc_impl::~torch_min_max_normalize_vcvc_impl() {}

int torch_min_max_normalize_vcvc_impl::work(
    int noutput_items, gr_vector_const_void_star &input_items,
    gr_vector_void_star &output_items) {

  input_type *in =
      reinterpret_cast<input_type *>(const_cast<void *>(input_items[0]));
  output_type *out = reinterpret_cast<output_type *>(output_items[0]);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

  // Interpret the in-buffer as a Tensor of type Float32 and transfer to the
  // whatever device it needs to be on.
  auto input = torch::from_blob(reinterpret_cast<void *>(in),
                                {noutput_items * 2}, options);
  input.to(d_device);

  // Perform min/max normailization (forces max/min values to be 1 and -1
  // respetively.)
  input -= input.min();
  input /= input.max() - input.min();
  input = 2 * input - 1;

  // Copy back to CPU
  input.to(torch::kCPU);

  // Copy the memory to the output buffer
  std::memcpy(out, input.contiguous().data_ptr(),
              noutput_items * sizeof(gr_complex));

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
