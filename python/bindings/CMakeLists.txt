# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

# #######################################################################
# Check if there is C++ code at all
# #######################################################################
if(NOT torchdsp_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT torchdsp_sources)

# #######################################################################
# Check for pygccxml
# #######################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
)

include(GrPybind)

# #######################################################################
# Python Bindings
# #######################################################################
list(APPEND torchdsp_python_files
    triton_block_python.cc
    triton_model_python.cc 
    triton_fir_filter_ff_python.cc
    python_bindings.cc)

GR_PYBIND_MAKE_OOT(torchdsp
    ../..
    gr::torchdsp
    "${torchdsp_python_files}")

install(TARGETS torchdsp_python DESTINATION ${GR_PYTHON_DIR}/torchdsp COMPONENT pythonapi)
