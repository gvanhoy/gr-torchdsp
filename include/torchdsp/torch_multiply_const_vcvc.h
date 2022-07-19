/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_MULTIPLY_CONST_VCVC_H
#define INCLUDED_TORCHDSP_MULTIPLY_CONST_VCVC_H

#include <gnuradio/sync_block.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API torch_multiply_const_vcvc : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<torch_multiply_const_vcvc> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of
     * torchdsp::torch_multiply_const_vcvc.
     *
     * To avoid accidental use of raw pointers, torchdsp::torch_multiply_const_vcvc's
     * constructor is in a private implementation
     * class. torchdsp::torch_multiply_const_vcvc::make is the public interface for
     * creating new instances.
     */
    static sptr make(const std::vector<gr_complex>& constant, unsigned int device_num);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_MULTIPLY_CONST_VCVC_H */
