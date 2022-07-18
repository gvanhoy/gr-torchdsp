/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TRITON_BLOCK_H
#define INCLUDED_TORCHDSP_TRITON_BLOCK_H

#include <gnuradio/sync_block.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API triton_block : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<triton_block> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of torchdsp::triton_block.
     *
     * To avoid accidental use of raw pointers, torchdsp::triton_block's
     * constructor is in a private implementation
     * class. torchdsp::triton_block::make is the public interface for
     * creating new instances.
     */
    static sptr make(
        const std::string& model_name,
        const size_t max_batch_size,
        const std::string& triton_url = "localhost:8000");
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_BLOCK_H */
