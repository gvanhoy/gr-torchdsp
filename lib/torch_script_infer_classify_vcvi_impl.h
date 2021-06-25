/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TORCH_SCRIPT_INFER_CLASSIFY_VCVI_IMPL_H
#define INCLUDED_TORCHDSP_TORCH_SCRIPT_INFER_CLASSIFY_VCVI_IMPL_H

#include <torch/script.h>
#include <torch/torch.h>
#include <torchdsp/torch_script_infer_classify_vcvi.h>

namespace gr {
namespace torchdsp {

class torch_script_infer_classify_vcvi_impl
    : public torch_script_infer_classify_vcvi {
private:
  torch::jit::script::Module d_model;
  torch::Device d_device;
  unsigned int d_device_num;
  unsigned int d_batch_size;
  unsigned int d_num_samples;

  torch::Tensor d_input;
  std::vector<torch::jit::IValue> d_inputs;

public:
  torch_script_infer_classify_vcvi_impl(std::string jit_model_path,
                                        unsigned int device_num,
                                        unsigned int batch_size,
                                        unsigned int num_samples);
  ~torch_script_infer_classify_vcvi_impl();

  // Where all the action really happens
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TORCH_SCRIPT_INFER_CLASSIFY_VCVI_IMPL_H */
