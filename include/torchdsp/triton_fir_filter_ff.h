/* -*- c++ -*- */
/*
 * Copyright 2022 gr-torchdsp author.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_TORCHDSP_TRITON_FIR_FILTER_FF_H
#define INCLUDED_TORCHDSP_TRITON_FIR_FILTER_FF_H

#include <gnuradio/sync_decimator.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API triton_fir_filter_ff : virtual public gr::sync_decimator
{
public:
    typedef std::shared_ptr<triton_fir_filter_ff> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of torchdsp::triton_fir_filter_ff.
     *
     * To avoid accidental use of raw pointers, torchdsp::triton_fir_filter_ff's
     * constructor is in a private implementation
     * class. torchdsp::triton_fir_filter_ff::make is the public interface for
     * creating new instances.
     */
    static sptr make(
        const std::string& model_name,
        const std::string& triton_url,
        unsigned int tap_size);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_FIR_FILTER_FF_H */
