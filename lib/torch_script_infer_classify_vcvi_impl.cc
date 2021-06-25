/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "torch_script_infer_classify_vcvi_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = int;

torch_script_infer_classify_vcvi::sptr torch_script_infer_classify_vcvi::make(
    std::string jit_model_path, unsigned int device_num,
    unsigned int batch_size, unsigned int num_samples) {
  return gnuradio::make_block_sptr<torch_script_infer_classify_vcvi_impl>(
      jit_model_path, device_num, batch_size, num_samples);
}

/*
 * The private constructor
 */
torch_script_infer_classify_vcvi_impl::torch_script_infer_classify_vcvi_impl(
    std::string jit_model_path, unsigned int device_num,
    unsigned int batch_size, unsigned int num_samples)
    : gr::sync_block("torch_script_infer_classify_vcvi",
                     gr::io_signature::make(
                         1, 1, sizeof(input_type) * batch_size * num_samples),
                     gr::io_signature::make(
                         1, 1, sizeof(output_type) * batch_size * num_samples)),
      d_batch_size(batch_size), d_device_num(device_num),
      d_num_samples(num_samples),
      d_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU,
               device_num) {
  try {
    d_model = torch::jit::load(jit_model_path.c_str());
    d_model.eval();
    d_model.to(d_device);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
  }
}

/*
 * Our virtual destructor.
 */
torch_script_infer_classify_vcvi_impl::
    ~torch_script_infer_classify_vcvi_impl() {}

int torch_script_infer_classify_vcvi_impl::work(
    int noutput_items, gr_vector_const_void_star &input_items,
    gr_vector_void_star &output_items) {

  input_type *in =
      reinterpret_cast<input_type *>(const_cast<void *>(input_items[0]));
  output_type *out = reinterpret_cast<output_type *>(output_items[0]);
  unsigned int items_processed = 0;

  while (items_processed < noutput_items) {

    // Interpret the in-buffer as a Tensor and transfer if to the GPU
    // We assume we're going to fit real/imaginary into floating-point chann
    d_input = torch::from_blob(in, {d_batch_size, d_num_samples, 2});
    d_input = d_input.to(d_device);

    // because we copied in the data channels-last
    d_input = d_input.permute({0, 2, 1});

    // We min-max normalize for each sample
    d_inputs.clear();
    d_inputs.push_back(d_input);

    auto result = d_model.forward(d_inputs)
                      .toTensor()
                      .argmax(1)
                      .to(torch::kInt32)
                      .to(torch::kCPU);

    std::memcpy(out, result.contiguous().data_ptr(), d_batch_size);

    in += d_num_samples * d_batch_size;
    out += d_num_samples * d_batch_size;
    items_processed += 1;
  }

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

} /* namespace torchdsp */
} /* namespace gr */
