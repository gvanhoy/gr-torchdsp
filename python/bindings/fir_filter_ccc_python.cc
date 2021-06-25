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
/* BINDTOOL_HEADER_FILE(fir_filter_ccc.h) */
/* BINDTOOL_HEADER_FILE_HASH(9227605d16718f24a634ebb4f5de8e72) */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <torchdsp/fir_filter_ccc.h>
// pydoc.h is automatically generated in the build directory
#include <fir_filter_ccc_pydoc.h>

void bind_fir_filter_ccc(py::module &m) {

  using fir_filter_ccc = gr::torchdsp::fir_filter_ccc;

  py::class_<fir_filter_ccc, gr::sync_decimator, gr::block, gr::basic_block,
             std::shared_ptr<fir_filter_ccc>>(m, "fir_filter_ccc",
                                              D(fir_filter_ccc))

      .def(py::init(&fir_filter_ccc::make), py::arg("taps"),
           py::arg("downsample_rate"), py::arg("device_num"),
           D(fir_filter_ccc, make))

      ;
}
