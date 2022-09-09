/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "triton_block_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace torchdsp {

using input_type = gr_complex;
using output_type = gr_complex;

triton_block::sptr triton_block::make(
    const std::string& model_name,
    const size_t max_batch_size,
    const std::string& triton_url,
    const std::vector<int>& input_sizes,
    const std::vector<int>& output_sizes) {
    auto model = triton_model::make(model_name, max_batch_size, triton_url);
    if (model == nullptr)
        throw std::runtime_error("Could not instantiate triton_model");


    std::cout << "Instantiated model" << std::endl;

    // We ask Triton for what the input signature is if one is not providided.
    // We sometimes need to provide one because complexf is not supported in Triton.
    return gnuradio::make_block_sptr<triton_block_impl>(
        model,
        input_sizes.size() == 0 ? model.get()->get_input_signature() : input_sizes,
        output_sizes.size() == 0 ? model.get()->get_output_signature() : output_sizes);
}

/*
 * The private constructor
 */
triton_block_impl::triton_block_impl(
    std::unique_ptr<triton_model>& model,
    const std::vector<int>& input_sizes,
    const std::vector<int>& output_sizes)
    : gr::sync_block(
          "triton_block",
          gr::io_signature::makev(1, -1, input_sizes),
          gr::io_signature::makev(1, -1, output_sizes)),
      model_(std::move(model)) // this is invoked after calling sync_block constructor.
{
    set_output_multiple(
        model_.get()->get_output_sizes()[0] / model_.get()->get_output_signature()[0]);
    std::cout << "Instantiated block" << std::endl;
}

/*
 * Our virtual destructor.
 */
triton_block_impl::~triton_block_impl() {}

int triton_block_impl::work(
    int noutput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items) {

    // std::cout << "Got " << noutput_items << " to produce." << std::endl;
    std::vector<const char*> in_ptrs;
    for (const auto& item : input_items)
        in_ptrs.push_back(static_cast<const char*>(item));

    std::vector<char*> out_ptrs;
    for (const auto& item : output_items)
        out_ptrs.push_back(static_cast<char*>(item));

    // num_items_per_patch is fixed.
    auto num_items_per_batch =
        model_.get()->get_output_sizes()[0] / model_.get()->get_output_signature()[0];
    auto batch_size = noutput_items / num_items_per_batch;

    model_->infer_batch(in_ptrs, out_ptrs, batch_size);

    return noutput_items;
}

} // namespace torchdsp
} /* namespace gr */
