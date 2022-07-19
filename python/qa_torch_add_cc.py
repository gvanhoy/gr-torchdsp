#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 gr-torchdsp author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks
import numpy as np
# from gnuradio import blocks
try:
    from torchdsp import torch_add_cc
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from torchdsp import torch_add_cc


class qa_torch_add_cc(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        instance = torch_add_cc(2, 0)

    def test_001_descriptive_test_name(self):
        taps = np.ones((12,), dtype=np.complex64) + 1.0j * \
            np.ones((12,), dtype=np.complex64)
        data = np.exp(1j*np.pi*np.linspace(0, 10000,
                      10001, endpoint=True)/2000)

        # Torch DSP
        source = blocks.vector_source_c(data, False, 1)
        source2 = blocks.vector_source_c(data, False, 1)
        add_block = torch_add_cc(2, 0)
        sink = blocks.vector_sink_c(1)

        self.tb.connect(source, add_block, sink)
        self.tb.connect(source2, (add_block, 1))
        self.tb.run()

        # Original GR
        self.tb2 = gr.top_block()
        source3 = blocks.vector_source_c(data, False, 1)
        source4 = blocks.vector_source_c(data, False, 1)
        add_block = blocks.torch_add_cc(1)
        sink2 = blocks.vector_sink_c(1)

        self.tb2.connect(source3, add_block, sink2)
        self.tb2.connect(source4, (add_block, 1))
        self.tb2.run()

        expected = np.asarray(sink2.data())
        output = np.asarray(sink.data())

        self.assertTrue(np.allclose(output, expected, atol=.1))


if __name__ == '__main__':
    gr_unittest.run(qa_torch_add_cc)
