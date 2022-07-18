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
    const std::string& triton_url) {
    auto model = triton_model::make(model_name, max_batch_size, triton_url);
    if (model == nullptr)
        throw std::runtime_error("Could not instantiate triton_model");

    std::cout << "Instantiated model" << std::endl;
    return gnuradio::make_block_sptr<triton_block_impl>(model);
}

/*
 * The private constructor
 */
triton_block_impl::triton_block_impl(std::unique_ptr<triton_model>& model)
    : gr::sync_block(
          "triton_block",
          gr::io_signature::makev(1, -1, model.get()->get_input_signature()),
          gr::io_signature::makev(1, -1, model.get()->get_output_signature())),
      model_(std::move(model)) // this is invoked after calling sync_block constructor.
{
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
