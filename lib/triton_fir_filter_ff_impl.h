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

#ifndef INCLUDED_TORCHDSP_TRITON_FIR_FILTER_FF_IMPL_H
#define INCLUDED_TORCHDSP_TRITON_FIR_FILTER_FF_IMPL_H

#include "shm_utils.h"
#include <http_client.h>
#include <torchdsp/triton_fir_filter_ff.h>
#include <torchdsp/triton_model.h>

namespace gr {
namespace torchdsp {

class triton_fir_filter_ff_impl : public triton_fir_filter_ff
{
private:
    std::unique_ptr<triton_model> model_;

public:
    triton_fir_filter_ff_impl(
        std::unique_ptr<triton_model>& model,
        unsigned int tap_size);
    ~triton_fir_filter_ff_impl();

    // Where all the action really happens
    int work(
        int noutput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_FIR_FILTER_FF_IMPL_H */
