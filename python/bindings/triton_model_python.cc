/*
 * Copyright 2022 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(triton_model.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(44d7fe109a40b372f1f20ac7c176949e)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <torchdsp/triton_model.h>
// pydoc.h is automatically generated in the build directory
#include <triton_model_pydoc.h>

void bind_triton_model(py::module& m) {

    using triton_model = ::gr::torchdsp::triton_model;


    py::class_<triton_model, std::shared_ptr<triton_model>>(
        m, "triton_model", D(triton_model))

        .def(
            py::init(&triton_model::make),
            py::arg("model_name"),
            py::arg("max_batch_size"),
            py::arg("triton_url") = "localhost:8000",
            D(triton_model, make))


        .def(
            "get_num_inputs",
            &triton_model::get_num_inputs,
            D(triton_model, get_num_inputs))


        .def(
            "get_num_outputs",
            &triton_model::get_num_outputs,
            D(triton_model, get_num_outputs))


        .def(
            "get_input_sizes",
            &triton_model::get_input_sizes,
            D(triton_model, get_input_sizes))


        .def(
            "get_output_sizes",
            &triton_model::get_output_sizes,
            D(triton_model, get_output_sizes))


        .def(
            "get_input_signature",
            &triton_model::get_input_signature,
            D(triton_model, get_input_signature))


        .def(
            "get_output_signature",
            &triton_model::get_output_signature,
            D(triton_model, get_output_signature))


        .def(
            "infer",
            &triton_model::infer,
            py::arg("in"),
            py::arg("out"),
            D(triton_model, infer))


        .def(
            "infer_batch",
            &triton_model::infer_batch,
            py::arg("in"),
            py::arg("out"),
            py::arg("batch_size"),
            D(triton_model, infer_batch))


        ;
}
