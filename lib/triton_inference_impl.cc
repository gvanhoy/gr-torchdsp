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

triton_inference::sptr triton_inference::make(std::string triton_url,
                                              std::string model_name,
                                              const std::vector<int64_t>& input_shape,
                                              const std::vector<int64_t>& output_shape)
{
    return gnuradio::make_block_sptr<triton_inference_impl>(
        triton_url, model_name, input_shape, output_shape);
}

int64_t triton_inference_impl::num_elements(const std::vector<int64_t>& shape)
{
    int64_t num_elements = 1;
    for (const int64_t& dim_size : shape)
        num_elements *= dim_size;

    return num_elements;
}

/*
 * The private constructor
 */
triton_inference_impl::triton_inference_impl(std::string triton_url,
                                             std::string model_name,
                                             const std::vector<int64_t>& input_shape,
                                             const std::vector<int64_t>& output_shape)
    : gr::sync_block(
          "triton_inference",
          gr::io_signature::make(1, 1, sizeof(input_type) * num_elements(input_shape)),
          gr::io_signature::make(1, 1, sizeof(output_type) * num_elements(output_shape))),
      input_byte_size_(sizeof(input_type) * num_elements(input_shape)),
      output_byte_size_(sizeof(output_type) * num_elements(output_shape)),
      input_shm_key_("/input_data_gr"),
      output_shm_key_("/output_data_gr"),
      options_(model_name)
{
    tc::Error err = tc::InferenceServerHttpClient::Create(&client_, triton_url, false);

    bool is_live;
    client_->IsServerLive(&is_live);
    if (!(err.IsOk() && is_live))
        GR_LOG_CRIT(d_debug_logger, "Could not connect to Triton Inference Server");

    bool is_ready;
    client_->IsServerReady(&is_ready);
    if (!(err.IsOk() && is_ready))
        GR_LOG_CRIT(d_debug_logger, "Could not connect to Triton Inference Server");

    std::string metadata;
    client_->ModelRepositoryIndex(&metadata);
    std::cout << metadata << std::endl;

    // In-case something is already registered.
    client_->UnregisterSystemSharedMemory();

    // Initialize the inputs with the data.
    tc::InferInput* input;
    tc::InferInput::Create(&input, "input__0", input_shape, "FP32");
    input_ptr_.reset(input);

    // Allocate system shared memory and register it with triton
    // The actual allocation of system shared memory should be refactored into a
    // custom_buffer class so that we just use our input buffer directly.

    // Also, we should actually just as the Triton Server for all of the model information
    // and set up buffers accordingly rather than forcing the user to input essentially
    // redundant information.
    int shm_fd_ip;
    tc::CreateSharedMemoryRegion(input_shm_key_, input_byte_size_, &shm_fd_ip);
    tc::MapSharedMemory(shm_fd_ip, 0, input_byte_size_, (void**)&triton_input_shm_);
    tc::CloseSharedMemory(shm_fd_ip);
    client_->RegisterSystemSharedMemory("input_data", input_shm_key_, input_byte_size_);
    input_ptr_->SetSharedMemory("input_data", input_byte_size_, 0 /* offset */);

    // Initialize outputs
    tc::InferRequestedOutput* output;
    tc::InferRequestedOutput::Create(&output, "output__0");
    output_ptr_.reset(output);

    // Allocation shared memory and register it.
    int shm_fd_op;
    tc::CreateSharedMemoryRegion(output_shm_key_, output_byte_size_, &shm_fd_op);
    tc::MapSharedMemory(shm_fd_op, 0, output_byte_size_, (void**)&triton_output_shm_);
    tc::CloseSharedMemory(shm_fd_op);
    client_->RegisterSystemSharedMemory(
        "output_data", output_shm_key_, output_byte_size_);
    output_ptr_->SetSharedMemory("output_data", output_byte_size_, 0 /* offset */);

    // Initialize InferRequest/Reply
    inputs_.push_back(input_ptr_.get());
    outputs_.push_back(output_ptr_.get());

    // Take latest model version
    options_.model_version_ = "";
}

/*
 * Our virtual destructor.
 */
triton_inference_impl::~triton_inference_impl()
{
    // Unregister shared memory
    client_->UnregisterSystemSharedMemory("input_data");
    client_->UnregisterSystemSharedMemory("output_data");

    // Cleanup shared memory
    tc::UnmapSharedMemory(triton_input_shm_, input_byte_size_);
    tc::UnlinkSharedMemoryRegion(input_shm_key_);
    tc::UnmapSharedMemory(triton_output_shm_, output_byte_size_);
    tc::UnlinkSharedMemoryRegion(output_shm_key_);
}

int triton_inference_impl::work(int noutput_items,
                                gr_vector_const_void_star& input_items,
                                gr_vector_void_star& output_items)
{
    bool is_live;

    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    for (unsigned int i = 0; i < noutput_items; i++) {
        // Copy input buffer to input shm
        std::memcpy(triton_input_shm_, in, input_byte_size_);

        // Request an inference
        client_->Infer(&results_, options_, inputs_, outputs_);

        // Copy output buffer to output shm
        std::memcpy(out, triton_output_shm_, output_byte_size_);

        // Increment pointers by byte_size // sizeof(gr_complex)
        in += input_byte_size_ >> 3;
        out += output_byte_size_ >> 3;
    }

    return noutput_items;
}

} // namespace torchdsp
} /* namespace gr */
