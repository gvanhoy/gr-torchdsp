/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_ADD_CC_IMPL_H
#define INCLUDED_TORCHDSP_ADD_CC_IMPL_H

#include <torch/torch.h>
#include <torchdsp/torch_add_cc.h>

namespace gr {
namespace torchdsp {

class torch_add_cc_impl : public torch_add_cc
{
private:
    unsigned int d_num_inputs;
    torch::Device d_device;

public:
    torch_add_cc_impl(unsigned int num_inputs, unsigned int device_num);
    ~torch_add_cc_impl();

    // Where all the action really happens
    int work(
        int noutput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_ADD_CC_IMPL_H */
