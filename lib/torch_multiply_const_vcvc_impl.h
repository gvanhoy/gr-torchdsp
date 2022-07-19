/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_MULTIPLY_CONST_VCVC_IMPL_H
#define INCLUDED_TORCHDSP_MULTIPLY_CONST_VCVC_IMPL_H

#include <torch/torch.h>
#include <torchdsp/torch_multiply_const_vcvc.h>

namespace gr {
namespace torchdsp {

class torch_multiply_const_vcvc_impl : public torch_multiply_const_vcvc
{
private:
    torch::Tensor d_constant_real;
    torch::Tensor d_constant_imag;
    torch::Device d_device;

public:
    torch_multiply_const_vcvc_impl(
        const std::vector<gr_complex>& constant,
        unsigned int device_num);
    ~torch_multiply_const_vcvc_impl();

    // Where all the action really happens
    int work(
        int noutput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_MULTIPLY_CONST_VCVC_IMPL_H */
