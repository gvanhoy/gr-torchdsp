/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_FFT_VCC_IMPL_H
#define INCLUDED_TORCHDSP_FFT_VCC_IMPL_H

#include <torch/torch.h>
#include <torchdsp/fft_vcc.h>

namespace gr {
namespace torchdsp {

class fft_vcc_impl : public fft_vcc
{
private:
    unsigned int d_fft_len;
    torch::Device d_device;

public:
    fft_vcc_impl(unsigned int fft_len, unsigned int device_num);
    ~fft_vcc_impl();

    // Where all the action really happens
    int work(int noutput_items, gr_vector_const_void_star& input_items, gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_FFT_VCC_IMPL_H */
