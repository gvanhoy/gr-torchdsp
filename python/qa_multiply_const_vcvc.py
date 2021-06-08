#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 gr-torchdsp author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks
import numpy as np

try:
    from torchdsp import multiply_const_vcvc
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from torchdsp import multiply_const_vcvc

class qa_multiply_const_vcvc(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        const = np.ones((12,), dtype=np.complex64) + 1.0j * np.ones((12,), dtype=np.complex64)
        instance = multiply_const_vcvc(const, 0)

    def test_001_multiply_const_sinusoid(self):
        const = np.ones((12,), dtype=np.complex64) + 1.0j * np.ones((12,), dtype=np.complex64)
        data = np.exp(1j*np.pi*np.linspace(0, 12000 - 1, 12000, endpoint=True)/2000)

        # Torch DSP
        source = blocks.vector_source_c(data, False, 12)
        multiply_const = multiply_const_vcvc(const, 0)
        sink = blocks.vector_sink_c(12)

        self.tb.connect(source, multiply_const, sink)
        self.tb.run()

        # Original GR 
        self.tb2 = gr.top_block()
        source2 = blocks.vector_source_c(data, False, 12)
        multiply_const = blocks.multiply_const_vcc(const)
        sink2 = blocks.vector_sink_c(12)

        self.tb2.connect(source2, multiply_const, sink2)
        self.tb2.run()

        expected = np.asarray(sink2.data())
        output = np.asarray(sink.data())

        self.assertTrue(np.allclose(output, expected, atol=.1))


if __name__ == '__main__':
    gr_unittest.run(qa_multiply_const_vcvc)
