/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/pybind11.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;

// Headers for binding functions
/**************************************/
// The following comment block is used for
// gr_modtool to insert function prototypes
// Please do not delete
/**************************************/
// BINDING_FUNCTION_PROTOTYPES(
// void bind_torch_fir_filter_ccc(py::module& m);
// void bind_torch_add_cc(py::module& m);
// void bind_torch_multiply_const_vcvc(py::module& m);
// void bind_torch_fft_vcc(py::module& m);
// void bind_torch_script_infer_classify_vcvi(py::module& m);
// void bind_torch_min_max_normalize_vcvc(py::module& m);
void bind_triton_block(py::module& m);
void bind_triton_model(py::module& m);
// ) END BINDING_FUNCTION_PROTOTYPES


// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy() {
    import_array();
    return NULL;
}

PYBIND11_MODULE(torchdsp_python, m) {
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    // Allow access to base block methods
    py::module::import("gnuradio.gr");

    /**************************************/
    // The following comment block is used for
    // gr_modtool to insert binding function calls
    // Please do not delete
    /**************************************/
    // BINDING_FUNCTION_CALLS(
    // bind_torch_fir_filter_ccc(m);
    // bind_torch_add_cc(m);
    // bind_torch_multiply_const_vcvc(m);
    // bind_torch_fft_vcc(m);
    // bind_torch_script_infer_classify_vcvi(m);
    // bind_torch_min_max_normalize_vcvc(m);
    bind_triton_block(m);
    bind_triton_model(m);
    // ) END BINDING_FUNCTION_CALLS
}