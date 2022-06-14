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

namespace tc = triton::client;


namespace gr {
namespace torchdsp {

class triton_inference_impl : public triton_inference
{
private:
    std::unique_ptr<tc::InferenceServerHttpClient> client_;
    size_t input_byte_size_;
    size_t output_byte_size_;
    std::shared_ptr<tc::InferInput> input_ptr_;
    std::shared_ptr<tc::InferRequestedOutput> output_ptr_;
    std::vector<tc::InferInput*> inputs_;
    std::vector<const tc::InferRequestedOutput*> outputs_;
    tc::InferResult* results_ = nullptr;
    tc::InferOptions options_;

    gr_complex* triton_input_shm_;
    gr_complex* triton_output_shm_;
    std::string input_shm_key_;
    std::string output_shm_key_;

public:
    triton_inference_impl(std::string triton_url,
                          std::string model_name,
                          const std::vector<int64_t>& input_shape,
                          const std::vector<int64_t>& output_shape);
    ~triton_inference_impl();
    static int64_t num_elements(const std::vector<int64_t>& shape);


    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TRITON_INFERENCE_IMPL_H */
