/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_fir_filter_ccc_IMPL_H
#define INCLUDED_TORCHDSP_fir_filter_ccc_IMPL_H

#include <torch/torch.h>
#include <torchdsp/fir_filter_ccc.h>

namespace gr {
namespace torchdsp {

class fir_filter_ccc_impl : public fir_filter_ccc
{
private:
    torch::Tensor d_real_taps;
    torch::Tensor d_imag_taps;
    torch::Tensor d_taps;
    torch::Device d_device;

public:
    fir_filter_ccc_impl(const std::vector<gr_complex>& taps, unsigned int device_num);
    ~fir_filter_ccc_impl();

    // Where all the action really happens
    int work(int noutput_items, gr_vector_const_void_star& input_items, gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_fir_filter_ccc_IMPL_H */
