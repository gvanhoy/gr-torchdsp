/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TRITON_MODEL_IMPL_H
#define INCLUDED_TORCHDSP_TRITON_MODEL_IMPL_H

#include "shm_utils.h"
#include <http_client.h>
#include <rapidjson/document.h>
#include <torchdsp/triton_model.h>

namespace tc = triton::client;

namespace gr {
namespace torchdsp {

class triton_model_impl : public triton_model
{
public:
    typedef struct io_metadata_s {
        std::vector<int64_t> shape;
        std::string name;
        std::string datatype;
    } io_metadata_t;

    typedef struct io_memory_s {
        size_t byte_size;
        std::string shm_key;
        void* data_ptr;
    } io_memory_t;

private:
    std::unique_ptr<tc::InferenceServerHttpClient> client_;
    std::string model_name_;
    std::vector<std::shared_ptr<tc::InferInput>> input_ptrs_;
    std::vector<std::shared_ptr<tc::InferRequestedOutput>> output_ptrs_;
    std::vector<io_memory_t> inputs_;
    std::vector<io_memory_t> outputs_;
    tc::InferResult* results_ = nullptr;
    tc::InferOptions options_;

public:
    triton_model_impl(const std::string& model_name, const std::string& triton_url);
    // triton_model_impl(triton_model_impl&& other);
    // triton_model_impl& operator=(triton_model_impl&& other);
    ~triton_model_impl();

    // Interface to GNU Radio Block
    int get_num_inputs() { return inputs_.size(); };
    int get_num_outputs() { return outputs_.size(); };
    std::vector<int> get_input_sizes() {
        std::vector<int> sizes;
        for (const auto& input : inputs_)
            sizes.push_back(input.byte_size);
        return sizes;
    }
    std::vector<int> get_output_sizes() {
        std::vector<int> sizes;
        for (const auto& output : outputs_)
            sizes.push_back(output.byte_size);
        return sizes;
    }
    void infer(std::vector<const char*> in_buffers, std::vector<char*> out_buffers);


    // Interface to Triton Server
    static rapidjson::Document get_model_metadata(
        const std::unique_ptr<tc::InferenceServerHttpClient>& client,
        const std::string& model_name);

    static bool
    is_server_healthy(const std::unique_ptr<tc::InferenceServerHttpClient>& client);

    void create_triton_input(
        const std::unique_ptr<tc::InferenceServerHttpClient>& client,
        const io_metadata_t& io_meta,
        const io_memory_t& io_mem);

    void create_triton_output(
        const std::unique_ptr<tc::InferenceServerHttpClient>& client,
        const io_metadata_t& io_meta,
        const io_memory_t& io_mem);

    // JSON Parsing
    static std::vector<int64_t> parse_io_shape(const rapidjson::Value& io_metadata) {
        std::vector<int64_t> shape;
        for (auto& dimension : io_metadata["shape"].GetArray())
            shape.push_back(dimension.GetInt64());

        return shape;
    };
    static std::string parse_io_name(const rapidjson::Value& io_metadata) {
        return io_metadata["name"].GetString();
    };
    static std::string parse_io_datatype(const rapidjson::Value& io_metadata) {
        return io_metadata["datatype"].GetString();
    };

    // Configuring IO
    static int64_t itemsize(const std::string& data_type);
    static int64_t num_elements(const std::vector<int64_t>& shape);
    io_memory_t allocate_shm(const io_metadata_t& io_meta);
};

} // namespace torchdsp
} // namespace gr
#endif /* INCLUDED_TORCHDSP_TRITON_MODEL_IMPL_H */
