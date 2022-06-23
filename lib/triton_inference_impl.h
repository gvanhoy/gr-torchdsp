/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TRITON_INFERENCE_IMPL_H
#define INCLUDED_TORCHDSP_TRITON_INFERENCE_IMPL_H

#include "shm_utils.h"
#include <http_client.h>
#include <torchdsp/triton_inference.h>
#include <torchdsp/triton_model.h>

namespace tc = triton::client;


namespace gr {
namespace torchdsp {

class triton_inference_impl : public triton_inference
{
private:
    std::unique_ptr<triton_model> model_;

public:
    triton_inference_impl(std::unique_ptr<triton_model>& model);
    ~triton_inference_impl();

    // Where all the action really happens
    int work(
        int noutput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_INFERENCE_IMPL_H */
