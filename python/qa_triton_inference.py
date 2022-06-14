#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Peraton Labs, Inc..
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
# from gnuradio import blocks
try:
    from torchdsp import triton_inference
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from torchdsp import triton_inference


class qa_triton_inference(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        instance = triton_inference("http://localhost:8001")


if __name__ == '__main__':
    gr_unittest.run(qa_triton_inference)
