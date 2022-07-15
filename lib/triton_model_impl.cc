/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "triton_model_impl.h"

namespace gr {
namespace torchdsp {

/**
 * @brief
 *
 * @param model_name
 * @param triton_url
 * @return triton_model::sptr
 */
triton_model::sptr
triton_model::make(const std::string& model_name, const std::string& triton_url) {
    return std::make_unique<triton_model_impl>(model_name, triton_url);
}

/**
 * @brief Move constructor. We use this to prevent memory leaks.
 *
 * @param other
 */
triton_model_impl::triton_model_impl(triton_model_impl&& other)
    : client_(std::move(other.client_)), options_(other.model_name_) {

    for (const auto& input_ptr : other.input_ptrs_)
        input_ptrs_.push_back(std::move(input_ptr));

    for (const auto& output_ptr : other.output_ptrs_)
        output_ptrs_.push_back(std::move(output_ptr));

    model_name_ = other.model_name_;
    inputs_ = other.inputs_;
    outputs_ = other.outputs_;
    results_ = other.results_;
}

/**
 * @brief Move assignment operator. We use this to prevent memory leaks.
 *
 * @param other
 * @return triton_model_impl&
 */
triton_model_impl& triton_model_impl::operator=(triton_model_impl&& other) {
    client_ = std::move(other.client_);
    options_.model_name_ = other.model_name_;

    std::cout << "Move constructor called" << std::endl;
    for (const auto& input_ptr : other.input_ptrs_)
        input_ptrs_.push_back(std::move(input_ptr));

    for (const auto& output_ptr : other.output_ptrs_)
        output_ptrs_.push_back(std::move(output_ptr));

    model_name_ = other.model_name_;
    inputs_ = other.inputs_;
    outputs_ = other.outputs_;
    results_ = other.results_;
    return *this;
}

/**
 * @brief Construct a new triton model impl::triton model impl object
 *
 * @param model_name model name available in TIS model repo
 * @param triton_url Non-protocol prefixed URL to running TIS instance
 */
triton_model_impl::triton_model_impl(
    const std::string& model_name,
    const std::string& triton_url)
    : model_name_(model_name), options_(model_name) {
    tc::Error err = tc::InferenceServerHttpClient::Create(&client_, triton_url, false);

    if (!triton_model_impl::is_server_healthy(client_)) {
        std::cerr << "Failed to connect to TIS instance at " << triton_url << std::endl;
        return;
    }

    std::cerr << "Server at " << triton_url << " determined to be healthy." << std::endl;

    auto model_metadata = triton_model_impl::get_model_metadata(client_, model_name);

    std::cerr << "Got model metadata for model " << model_name << std::endl;

    // Configure Inputs
    for (const auto& metadata : model_metadata["inputs"].GetArray()) {
        io_metadata_t io_meta = io_metadata_t{ parse_io_shape(metadata),
                                               parse_io_name(metadata),
                                               parse_io_datatype(metadata) };
        auto io_mem = allocate_shm(io_meta);
        inputs_.push_back(io_mem);
        create_triton_input(client_, io_meta, io_mem);
    }

    // std::cout << "Allocated input memory" << std::endl;

    // Configure Outputs
    for (const auto& metadata : model_metadata["outputs"].GetArray()) {
        io_metadata_t io_meta = io_metadata_t{ parse_io_shape(metadata),
                                               parse_io_name(metadata),
                                               parse_io_datatype(metadata) };
        auto io_mem = allocate_shm(io_meta);
        outputs_.push_back(io_mem);
        create_triton_output(client_, io_meta, io_mem);
    }

    // std::cout << "Allocated output memory" << std::endl;

    // std::cout << "Got " << this->get_num_inputs() << " inputs." << std::endl;
    // std::cout << "Got " << this->get_num_outputs() << " outputs." << std::endl;
    // std::cout << "Got [";
    // for (const auto& dim : this->get_input_sizes())
    //     std::cout << dim << ", ";
    // std::cout << "] for input sizes." << std::endl;

    // std::cout << "Got [";
    // for (const auto& dim : this->get_output_sizes())
    //     std::cout << dim << ", ";
    // std::cout << "] for output sizes." << std::endl;

    options_.model_version_ = "";
}


/**
 * @brief Destroy the triton model impl::triton model impl object
 *
 */
triton_model_impl::~triton_model_impl() {
    for (const auto& input : inputs_) {
        auto input_name = std::string("input_") + input.shm_key.substr(4);
        client_->UnregisterSystemSharedMemory(input_name);
        tc::UnmapSharedMemory(input.data_ptr, input.byte_size);
        tc::UnlinkSharedMemoryRegion(input.shm_key);
    }

    for (const auto& output : outputs_) {
        auto output_name = std::string("output_") + output.shm_key.substr(4);
        client_->UnregisterSystemSharedMemory("output_data");
        tc::UnmapSharedMemory(output.data_ptr, output.byte_size);
        tc::UnlinkSharedMemoryRegion(output.shm_key);
    }
}


/**
 * @brief
 *
 * @param client
 * @return true server is ready and live
 * @return false server ain't ready or ain't live
 */
bool triton_model_impl::is_server_healthy(
    const std::unique_ptr<tc::InferenceServerHttpClient>& client) {
    bool is_live;
    tc::Error err = client->IsServerLive(&is_live);
    if (!(err.IsOk() && is_live))
        std::cerr << "Could not connect to Triton Inference Server -- Server not live"
                  << std::endl;

    std::cerr << err.Message() << std::endl;

    bool is_ready;
    err = client->IsServerReady(&is_ready);
    if (!(err.IsOk() && is_ready))
        std::cerr << "Could not connect to Triton Inference Server -- Server not ready"
                  << std::endl;

    return is_ready & is_live;
}

/**
 * @brief
 *
 * @param shape
 * @return int64_t
 */
int64_t triton_model_impl::num_elements(const std::vector<int64_t>& shape) {
    int64_t num_elements = 1;
    for (const int64_t& dim_size : shape)
        num_elements *= dim_size;

    return num_elements;
}

/**
 * @brief
 *
 * @param data_type
 * @return int64_t
 */
int64_t triton_model_impl::itemsize(const std::string& data_type) {
    if (std ::string("FP32").compare(data_type) == 0)
        return 4;
    if (std ::string("FLOAT32").compare(data_type) == 0)
        return 4;
    if (std ::string("INT32").compare(data_type) == 0)
        return 4;
    if (std ::string("INT16").compare(data_type) == 0)
        return 2;
    if (std ::string("INT8").compare(data_type) == 0)
        return 1;

    return 0;
}

/**
 * @brief
 *
 * @param client
 * @param model_name
 * @return rapidjson::Document
 */
rapidjson::Document triton_model_impl::get_model_metadata(
    const std::unique_ptr<tc::InferenceServerHttpClient>& client,
    const std::string& model_name) {
    std::string model_metadata;
    client->ModelMetadata(&model_metadata, model_name);
    rapidjson::Document json_metadata;
    json_metadata.Parse(model_metadata.c_str());
    return json_metadata;
}

/**
 * @brief
 *
 * @param io_meta
 * @return triton_model_impl::io_memory_t
 */
triton_model_impl::io_memory_t
triton_model_impl::allocate_shm(const io_metadata_t& io_meta) {
    int shm_fd_ip;
    void* data_ptr;
    std::string shm_key = std::string("/gr_") + io_meta.name;
    size_t byte_size = num_elements(io_meta.shape) * itemsize(io_meta.datatype);

    tc::CreateSharedMemoryRegion(shm_key, byte_size, &shm_fd_ip);
    tc::MapSharedMemory(shm_fd_ip, 0, byte_size, (void**)&data_ptr);
    tc::CloseSharedMemory(shm_fd_ip);

    io_memory_t io_mem{ byte_size, shm_key, data_ptr };
    return io_mem;
}

/**
 * @brief
 *
 * @param client
 * @param io_meta
 * @param io_mem
 */
void triton_model_impl::create_triton_input(
    const std::unique_ptr<tc::InferenceServerHttpClient>& client,
    const io_metadata_t& io_meta,
    const io_memory_t& io_mem) {
    // Create shared ptr for the InferInput
    tc::InferInput* input;

    std::shared_ptr<tc::InferInput> input_ptr;
    tc::InferInput::Create(&input, io_meta.name, io_meta.shape, io_meta.datatype);
    input_ptr.reset(input);
    input_ptrs_.push_back(input_ptr);

    // Inform TIS about the memory
    client->RegisterSystemSharedMemory(
        std::string("input_") + io_meta.name, io_mem.shm_key, io_mem.byte_size);

    input_ptrs_.back()->SetSharedMemory(
        std::string("input_") + io_meta.name, io_mem.byte_size, 0);
}

/**
 * @brief
 *
 * @param client
 * @param io_meta
 * @param io_mem
 */
void triton_model_impl::create_triton_output(
    const std::unique_ptr<tc::InferenceServerHttpClient>& client,
    const io_metadata_t& io_meta,
    const io_memory_t& io_mem) {
    // Create shared ptr for the InferRequestedOutput
    tc::InferRequestedOutput* output;
    std::shared_ptr<tc::InferRequestedOutput> output_ptr;
    tc::InferRequestedOutput::Create(&output, io_meta.name);
    output_ptr.reset(output);
    output_ptrs_.push_back(output_ptr);

    // Inform TIS about the memory
    client->RegisterSystemSharedMemory(
        std::string("output_") + io_meta.name, io_mem.shm_key, io_mem.byte_size);

    output_ptrs_.back()->SetSharedMemory(
        std::string("output_") + io_meta.name, io_mem.byte_size, 0);
}

void triton_model_impl::infer(
    std::vector<const char*> in_buffers,
    std::vector<char*> out_buffers) {
    // it'd be great if we can avoid this, but really may not be necessary to avoid
    for (uint16_t idx = 0; idx < in_buffers.size(); idx++)
        std::memcpy(inputs_[idx].data_ptr, in_buffers[idx], inputs_[idx].byte_size);

    // std::cout << "Copied to input buffers." << std::endl;
    std::vector<tc::InferInput*> inputs;
    for (const auto& input_ptr : input_ptrs_)
        inputs.push_back(input_ptr.get());

    std::vector<const tc::InferRequestedOutput*> outputs;
    for (const auto& output_ptr : output_ptrs_)
        outputs.push_back(output_ptr.get());

    // std::cout << "Copied input and output raw pointers." << std::endl;
    tc::InferResult* result;
    client_->Infer(&result, options_, inputs, outputs);

    // std::cout << "Inferred on result" << std::endl;
    // it'd be great if we can avoid this, but really may not be necessary to avoid
    for (uint16_t idx = 0; idx < out_buffers.size(); idx++)
        std::memcpy(out_buffers[idx], outputs_[idx].data_ptr, outputs_[idx].byte_size);
}


} // namespace torchdsp
} // namespace gr
