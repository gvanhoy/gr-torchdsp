/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_FFT_VCC_H
#define INCLUDED_TORCHDSP_FFT_VCC_H

#include <gnuradio/sync_block.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API torch_fft_vcc : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<torch_fft_vcc> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of torchdsp::torch_fft_vcc.
     *
     * To avoid accidental use of raw pointers, torchdsp::torch_fft_vcc's
     * constructor is in a private implementation
     * class. torchdsp::torch_fft_vcc::make is the public interface for
     * creating new instances.
     */
    static sptr make(unsigned int fft_len, unsigned int device_num);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_FFT_VCC_H */
