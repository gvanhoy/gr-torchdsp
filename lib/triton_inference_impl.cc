/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "triton_inference_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;

triton_inference::sptr
triton_inference::make(const std::string& model_name, const std::string& triton_url) {
    auto model = triton_model::make(model_name, triton_url);
    return gnuradio::make_block_sptr<triton_inference_impl>(model);
}

/*
 * The private constructor
 */
triton_inference_impl::triton_inference_impl(std::unique_ptr<triton_model>& model)
    : gr::sync_block(
          "triton_inference",
          gr::io_signature::makev(1, -1, model.get()->get_input_sizes()),
          gr::io_signature::makev(1, -1, model.get()->get_output_sizes())),
      model_(std::move(model)) // this is invoked after calling sync_block constructor.
{}

/*
 * Our virtual destructor.
 */
triton_inference_impl::~triton_inference_impl() {}

int triton_inference_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items) {

    std::vector<const char*> in_ptrs;
    for (const auto& item : input_items)
        in_ptrs.push_back(static_cast<const char*>(item));

    std::vector<char*> out_ptrs;
    for (const auto& item : output_items)
        out_ptrs.push_back(static_cast<char*>(item));

    for (unsigned int i = 0; i < noutput_items; i++) {
        model_->infer(in_ptrs, out_ptrs);

        // Increment pointers by byte_size // sizeof(gr_complex)
        for (unsigned int j = 0; j < in_ptrs.size(); j++)
            in_ptrs[j] += model_.get()->get_input_sizes()[j];

        for (unsigned int j = 0; j < out_ptrs.size(); j++)
            out_ptrs[j] += model_.get()->get_output_sizes()[j];
    }

    return noutput_items;
}

} // namespace torchdsp
} /* namespace gr */
