/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TORCH_MIN_MAX_NORMALIZE_VCVC_IMPL_H
#define INCLUDED_TORCHDSP_TORCH_MIN_MAX_NORMALIZE_VCVC_IMPL_H

#include <torch/torch.h>
#include <torchdsp/torch_min_max_normalize_vcvc.h>

namespace gr {
namespace torchdsp {

class torch_min_max_normalize_vcvc_impl : public torch_min_max_normalize_vcvc {
private:
  torch::Device d_device;
  unsigned int d_device_num;

public:
  torch_min_max_normalize_vcvc_impl(unsigned int device_num);
  ~torch_min_max_normalize_vcvc_impl();

  // Where all the action really happens
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TORCH_MIN_MAX_NORMALIZE_VCVC_IMPL_H */
