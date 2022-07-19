#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 gr-torchdsp author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks
import numpy as np
import timeit

# from gnuradio import blocks
try:
    from torchdsp import torch_fft_vcc
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from torchdsp import torch_fft_vcc


class qa_torch_fft_vcc(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        instance = torch_fft_vcc(256, 0)

    def test_001_fft_sinusoid(self):
        data = np.exp(1j*np.pi*np.linspace(0, 2560000 - 1,
                      2560000, endpoint=True)/256).astype(np.complex64)

        source = blocks.vector_source_c(data, False, 256)
        fft_block = torch_fft_vcc(256, 0)
        sink = blocks.vector_sink_c(256)

        self.tb.connect(source, fft_block, sink)
        self.tb.run()

        expected = np.fft.fft(data[:256])
        output = np.asarray(sink.data()[:256])

        self.assertTrue(np.allclose(output, expected, atol=.1))


if __name__ == '__main__':
    gr_unittest.run(qa_torch_fft_vcc)
