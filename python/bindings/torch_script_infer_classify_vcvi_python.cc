/*
 * Copyright 2021 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually
 * edited  */
/* The following lines can be configured to regenerate this file during cmake */
/* If manual edits are made, the following tags should be modified accordingly.
 */
/* BINDTOOL_GEN_AUTOMATIC(0) */
/* BINDTOOL_USE_PYGCCXML(0) */
/* BINDTOOL_HEADER_FILE(torch_script_infer_classify_vcvi.h) */
/* BINDTOOL_HEADER_FILE_HASH(df3e466cf1f2bbfa5f56dabfe03a3fc1) */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <torchdsp/torch_script_infer_classify_vcvi.h>
// pydoc.h is automatically generated in the build directory
#include <torch_script_infer_classify_vcvi_pydoc.h>

void bind_torch_script_infer_classify_vcvi(py::module &m) {

  using torch_script_infer_classify_vcvi =
      gr::torchdsp::torch_script_infer_classify_vcvi;

  py::class_<torch_script_infer_classify_vcvi, gr::sync_block, gr::block,
             gr::basic_block,
             std::shared_ptr<torch_script_infer_classify_vcvi>>(
      m, "torch_script_infer_classify_vcvi",
      D(torch_script_infer_classify_vcvi))

      .def(py::init(&torch_script_infer_classify_vcvi::make),
           py::arg("jit_model_path"), py::arg("device_num"),
           py::arg("batch_size"), py::arg("num_samples"),
           D(torch_script_infer_classify_vcvi, make));
}
