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
triton_model::sptr triton_model::make(
    const std::string& model_name,
    const size_t max_batch_size,
    const std::string& triton_url) {
    auto model =
        std::make_unique<triton_model_impl>(model_name, max_batch_size, triton_url);

    if (model.get()->get_num_inputs() == 0) {
        model.reset();
        return nullptr;
    }

    return model;
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
    const size_t max_batch_size,
    const std::string& triton_url)
    : model_name_(model_name), max_batch_size_(max_batch_size), options_(model_name) {
    tc::Error err = tc::InferenceServerHttpClient::Create(&client_, triton_url, false);

    if (!triton_model_impl::is_server_healthy(client_)) {
        std::cerr << "Failed to connect to TIS instance at " << triton_url << std::endl;
        return;
    }

    std::cout << "Server at " << triton_url << " determined to be healthy." << std::endl;

    auto model_metadata = triton_model_impl::get_model_metadata(client_, model_name);
    std::cout << "Got model metadata for model " << model_name << std::endl;

    // Configure Inputs
    for (const auto& metadata : model_metadata["inputs"].GetArray()) {
        io_metadata_t io_meta = io_metadata_t{ parse_io_shape(metadata),
                                               parse_io_name(metadata),
                                               parse_io_datatype(metadata) };
        auto io_mem = allocate_shm(io_meta, max_batch_size_);
        if (io_mem.element_byte_size == 0)
            break;
        inputs_.push_back(io_mem);
        create_triton_input(client_, io_meta, io_mem);
    }

    // std::cout << "Allocated input memory" << std::endl;

    // Configure Outputs
    for (const auto& metadata : model_metadata["outputs"].GetArray()) {
        io_metadata_t io_meta = io_metadata_t{ parse_io_shape(metadata),
                                               parse_io_name(metadata),
                                               parse_io_datatype(metadata) };
        auto io_mem = allocate_shm(io_meta, max_batch_size_);
        if (io_mem.element_byte_size == 0)
            break;
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
    int idx = 0;
    for (const auto& input : inputs_) {
        client_->UnregisterSystemSharedMemory(registered_input_names_[idx]);
        tc::UnmapSharedMemory(input.data_ptr, input.element_byte_size * input.batch_size);
        tc::UnlinkSharedMemoryRegion(input.shm_key);
        idx++;
    }

    idx = 0;
    for (const auto& output : outputs_) {
        client_->UnregisterSystemSharedMemory(registered_output_names_[idx]);
        tc::UnmapSharedMemory(
            output.data_ptr, output.element_byte_size * output.batch_size);
        tc::UnlinkSharedMemoryRegion(output.shm_key);
        idx++;
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
triton_model_impl::io_memory_t triton_model_impl::allocate_shm(
    const io_metadata_t& io_meta,
    const size_t max_batch_size) {

    static unsigned int segment_number = 0;

    int shm_fd_ip;
    void* data_ptr;
    std::string shm_key =
        std::string("/gr_") + io_meta.name + std::to_string(segment_number);
    size_t num_bytes =
        num_elements(io_meta.shape) * itemsize(io_meta.datatype) * max_batch_size;

    tc::Error error = tc::CreateSharedMemoryRegion(shm_key, num_bytes, &shm_fd_ip);
    tc::MapSharedMemory(shm_fd_ip, 0, num_bytes, (void**)&data_ptr);
    tc::CloseSharedMemory(shm_fd_ip);

    if (!error.IsOk())
        return io_memory_t{ 0, 0, 0, "", 0 };

    segment_number += 1;
    return io_memory_t{ static_cast<size_t>(itemsize(io_meta.datatype)),
                        num_bytes / max_batch_size,
                        max_batch_size,
                        shm_key,
                        data_ptr };
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

    static int input_number = 0;
    // Create shared ptr for the InferInput
    tc::InferInput* input;

    std::shared_ptr<tc::InferInput> input_ptr;
    tc::InferInput::Create(&input, io_meta.name, io_meta.shape, io_meta.datatype);
    input_ptr.reset(input);
    input_ptrs_.push_back(input_ptr);

    // Inform TIS about the memory
    auto registered_name =
        std::to_string(input_number) + std::string("input_") + io_meta.name;
    registered_input_names_.push_back(registered_name);

    client->RegisterSystemSharedMemory(
        registered_name, io_mem.shm_key, io_mem.element_byte_size * io_mem.batch_size);

    input_ptrs_.back()->SetSharedMemory(
        registered_name, io_mem.element_byte_size * io_mem.batch_size, 0);

    input_number += 1;
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

    static int output_number = 0;

    // Create shared ptr for the InferRequestedOutput
    tc::InferRequestedOutput* output;
    std::shared_ptr<tc::InferRequestedOutput> output_ptr;
    tc::InferRequestedOutput::Create(&output, io_meta.name);
    output_ptr.reset(output);
    output_ptrs_.push_back(output_ptr);

    // Inform TIS about the memory
    auto registered_name =
        std::to_string(output_number) + std::string("output_") + io_meta.name;
    registered_output_names_.push_back(registered_name);
    client->RegisterSystemSharedMemory(
        std::to_string(output_number) + std::string("output_") + io_meta.name,
        io_mem.shm_key,
        io_mem.element_byte_size * io_mem.batch_size);

    output_ptrs_.back()->SetSharedMemory(
        std::to_string(output_number) + std::string("output_") + io_meta.name,
        io_mem.element_byte_size * io_mem.batch_size,
        0);

    output_number += 1;
}

void triton_model_impl::infer(
    std::vector<const char*> in_buffers,
    std::vector<char*> out_buffers) {
    // it'd be great if we can avoid this, but really may not be necessary to avoid
    for (uint16_t idx = 0; idx < in_buffers.size(); idx++)
        std::memcpy(
            inputs_[idx].data_ptr, in_buffers[idx], inputs_[idx].element_byte_size);

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
        std::memcpy(
            out_buffers[idx], outputs_[idx].data_ptr, outputs_[idx].element_byte_size);
}

void triton_model_impl::infer_batch(
    std::vector<const char*> in_buffers,
    std::vector<char*> out_buffers,
    size_t batch_size) {

    for (uint16_t idx = 0; idx < in_buffers.size(); idx++)
        std::memcpy(
            inputs_[idx].data_ptr,
            in_buffers[idx],
            inputs_[idx].element_byte_size * batch_size);

    std::vector<tc::InferInput*> inputs;
    for (const auto& input_ptr : input_ptrs_) {
        // We have to modify the shape of the input
        std::vector<int64_t> new_shape;
        for (const auto& dim : input_ptr->Shape())
            new_shape.push_back(dim);
        new_shape[0] = batch_size; // We just override the first dimension.
        input_ptr->SetShape(new_shape);
        inputs.push_back(input_ptr.get());
    }

    std::vector<const tc::InferRequestedOutput*> outputs;
    for (const auto& output_ptr : output_ptrs_) {
        outputs.push_back(output_ptr.get());
    }

    tc::InferResult* result;
    client_->Infer(&result, options_, inputs, outputs);

    // std::cout << "Inferred on result" << std::endl;
    // it'd be great if we can avoid this, but really may not be necessary to avoid
    for (uint16_t idx = 0; idx < out_buffers.size(); idx++)
        std::memcpy(
            out_buffers[idx],
            outputs_[idx].data_ptr,
            outputs_[idx].element_byte_size * batch_size);
}

} // namespace torchdsp
} // namespace gr
