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
/* BINDTOOL_HEADER_FILE(torch_add_cc.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(db613d15509466fb2974f38cbff4b501)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <torchdsp/torch_add_cc.h>
// pydoc.h is automatically generated in the build directory
#include <torch_add_cc_pydoc.h>

void bind_torch_add_cc(py::module& m)
{

    using torch_add_cc    = ::gr::torchdsp::torch_add_cc;


    py::class_<torch_add_cc, gr::sync_block, gr::block, gr::basic_block,
        std::shared_ptr<torch_add_cc>>(m, "torch_add_cc", D(torch_add_cc))

        .def(py::init(&torch_add_cc::make),
           py::arg("num_inputs"),
           py::arg("device_num"),
           D(torch_add_cc,make)
        )
        



        ;




}








