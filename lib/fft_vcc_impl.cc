/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "fft_vcc_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;

fft_vcc::sptr fft_vcc::make(unsigned int fft_len, unsigned int device_num) {
  return gnuradio::make_block_sptr<fft_vcc_impl>(fft_len, device_num);
}

/*
 * The private constructor
 */
fft_vcc_impl::fft_vcc_impl(unsigned int fft_len, unsigned int device_num)
    : gr::sync_block(
          "fft_vcc", gr::io_signature::make(1, 1, sizeof(input_type) * fft_len),
          gr::io_signature::make(1, 1, sizeof(output_type) * fft_len)),
      d_fft_len(fft_len),
      d_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU,
               device_num) {}

/*
 * Our virtual destructor.
 */
fft_vcc_impl::~fft_vcc_impl() {}

int fft_vcc_impl::work(int noutput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items) {
  input_type *in =
      reinterpret_cast<input_type *>(const_cast<void *>(input_items[0]));
  output_type *out = reinterpret_cast<output_type *>(output_items[0]);

  auto options =
      torch::TensorOptions().dtype(torch::kComplexFloat).requires_grad(false);

  for (unsigned int idx = 0; idx < noutput_items; idx++) {
    // Interpret the in-buffer as a Tensor of type Float32 and transfer to the
    // whatever device it needs to be on.
    auto input =
        torch::from_blob(reinterpret_cast<void *>(in), {d_fft_len}, options);
    input.to(d_device);

    // Do the FFT and copy out to CPU
    auto result = torch::fft::fftshift(torch::fft::fft(input)).to(torch::kCPU);

    // Copy result tensor to output buffer
    std::memcpy(out, result.contiguous().data_ptr(), d_fft_len * sizeof(gr_complex));

    // Increment pointers.
    in += d_fft_len;
    out += d_fft_len;
  }

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
