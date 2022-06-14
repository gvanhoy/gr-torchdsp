/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TRITON_INFERENCE_H
#define INCLUDED_TORCHDSP_TRITON_INFERENCE_H

#include <gnuradio/sync_block.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API triton_inference : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<triton_inference> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of torchdsp::triton_inference.
     *
     * To avoid accidental use of raw pointers, torchdsp::triton_inference's
     * constructor is in a private implementation
     * class. torchdsp::triton_inference::make is the public interface for
     * creating new instances.
     */
    static sptr make(std::string triton_url,
                     std::string model_name,
                     const std::vector<int64_t>& input_shape,
                     const std::vector<int64_t>& output_shape);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_INFERENCE_H */
