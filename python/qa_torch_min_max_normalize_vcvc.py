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
    from torchdsp import torch_min_max_normalize_vcvc
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from torchdsp import torch_min_max_normalize_vcvc

class qa_torch_min_max_normalize_vcvc(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        instance = torch_min_max_normalize_vcvc(0)

    def test_001_descriptive_test_name(self):
        data = 10*np.exp(1j*np.pi*np.linspace(0, 10000, 10001, endpoint=True)/2000).astype(np.complex64)

        # Torch DSP
        source = blocks.vector_source_c(data, False, 1)
        min_max = torch_min_max_normalize_vcvc(0)
        sink = blocks.vector_sink_c(1)

        self.tb.connect(source, min_max, sink)
        self.tb.run()

        # Regular NumPy
        expected = data.view(np.float32)
        expected -= expected.min()
        expected /= expected.max() - expected.min()
        expected = 2*expected - 1
        expected = expected.view(np.complex64)
        output = np.asarray(sink.data())

        self.assertTrue(np.allclose(output, expected, atol=.1))


if __name__ == '__main__':
    gr_unittest.run(qa_torch_min_max_normalize_vcvc)
