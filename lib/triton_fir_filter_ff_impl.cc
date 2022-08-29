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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "triton_fir_filter_ff_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

triton_fir_filter_ff::sptr triton_fir_filter_ff::make(
    const std::string& model_name,
    const std::string& triton_url,
    unsigned int tap_size) {
    auto model = triton_model::make(model_name, 256, triton_url);

    if (model == nullptr)
        throw std::runtime_error("Could not instantiate triton_model");

    return gnuradio::make_block_sptr<triton_fir_filter_ff_impl>(model, tap_size);
}


/*
 * The private constructor
 */
triton_fir_filter_ff_impl::triton_fir_filter_ff_impl(
    std::unique_ptr<triton_model>& model,
    unsigned int tap_size)
    : gr::sync_decimator(
          "triton_fir_filter_ff",
          gr::io_signature::make(1, 1, sizeof(float)),
          gr::io_signature::make(1, 1, sizeof(float)),
          1),
      model_(std::move(model)) {
    set_output_multiple(1024); // hard-coded from config.pbtxt
    set_history(tap_size);     // should come from exported model's taps in make_model.py
}

/*
 * Our virtual destructor.
 */
triton_fir_filter_ff_impl::~triton_fir_filter_ff_impl() {}

int triton_fir_filter_ff_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items) {

    std::vector<const char*> in_ptrs;
    in_ptrs.push_back(static_cast<const char*>(input_items[0]));

    std::vector<char*> out_ptrs;
    out_ptrs.push_back(static_cast<char*>(output_items[0]));

    // num_items_per_patch is fixed.
    auto batch_size = noutput_items / this->output_multiple();
    model_->infer_batch(in_ptrs, out_ptrs, batch_size);

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} // namespace torchdsp
} /* namespace gr */
