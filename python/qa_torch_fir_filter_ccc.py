#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 gr-torchdsp author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, filter
from gnuradio import blocks
import numpy as np

# from gnuradio import blocks
try:
    from torchdsp import torch_fir_filter_ccc
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from torchdsp import torch_fir_filter_ccc


class qa_torch_fir_filter_ccc(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        taps = np.ones((12,), dtype=np.complex64) + 1j * \
            np.ones((12,), dtype=np.complex64)
        instance = torch_fir_filter_ccc(taps, 1, 0)

    def test_001_filter_matches_gnuradio_fir(self):
        taps = np.ones((12,), dtype=np.complex64) + 1.0j * \
            np.ones((12,), dtype=np.complex64)
        data = np.exp(1j*np.pi*np.linspace(0, 10000,
                      10001, endpoint=True)/2000)

        source = blocks.vector_source_c(data, False, 1)
        filter_block = torch_fir_filter_ccc(taps, 1, 0)
        sink = blocks.vector_sink_c(1)

        self.tb.connect(source, filter_block, sink)
        self.tb.run()

        self.tb2 = gr.top_block()
        source2 = blocks.vector_source_c(data, False, 1)
        filter_block = filter.torch_fir_filter_ccc(1, taps)
        sink2 = blocks.vector_sink_c(1)

        self.tb2.connect(source2, filter_block, sink2)
        self.tb2.run()

        expected = np.asarray(sink2.data())
        output = np.asarray(sink.data())

        self.assertTrue(np.allclose(output, expected, atol=.1))

    def test_002_downsample_filter_matches_gnuradio_fir(self):
        taps = np.ones((12,), dtype=np.complex64) + 1.0j * \
            np.ones((12,), dtype=np.complex64)
        data = np.exp(1j*np.pi*np.linspace(0, 10000,
                      10001, endpoint=True)/2000)

        source = blocks.vector_source_c(data, False, 1)
        filter_block = torch_fir_filter_ccc(taps, 2, 0)
        sink = blocks.vector_sink_c(1)

        self.tb.connect(source, filter_block, sink)
        self.tb.run()

        self.tb2 = gr.top_block()
        source2 = blocks.vector_source_c(data, False, 1)
        filter_block = filter.torch_fir_filter_ccc(2, taps)
        sink2 = blocks.vector_sink_c(1)

        self.tb2.connect(source2, filter_block, sink2)
        self.tb2.run()

        expected = np.asarray(sink2.data())
        output = np.asarray(sink.data())

        self.assertTrue(np.allclose(output, expected, atol=.1))

    def test_003_downsample_by_3_filter_matches_gnuradio_fir(self):
        taps = np.ones((12,), dtype=np.complex64) + 1.0j * \
            np.ones((12,), dtype=np.complex64)
        data = np.exp(1j*np.pi*np.linspace(0, 10000,
                      10001, endpoint=True)/2000)

        source = blocks.vector_source_c(data, False, 1)
        filter_block = torch_fir_filter_ccc(taps, 3, 0)
        sink = blocks.vector_sink_c(1)

        self.tb.connect(source, filter_block, sink)
        self.tb.run()

        self.tb2 = gr.top_block()
        source2 = blocks.vector_source_c(data, False, 1)
        filter_block = filter.torch_fir_filter_ccc(3, taps)
        sink2 = blocks.vector_sink_c(1)

        self.tb2.connect(source2, filter_block, sink2)
        self.tb2.run()

        expected = np.asarray(sink2.data())
        output = np.asarray(sink.data())

        self.assertTrue(np.allclose(output, expected, atol=.1))


if __name__ == '__main__':
    gr_unittest.run(qa_torch_fir_filter_ccc)
