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
    from torchdsp import torch_script_infer_classify_vcvi
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from torchdsp import torch_script_infer_classify_vcvi


class qa_torch_script_infer_classify_vcvi(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    # These only work if a trained model is included. I didn't want to include
    # a huge weights file into this repo.
    def test_instance(self):
        pass
        # instance = torch_script_infer_classify_vcvi("vt_cnn2.pt", 0, 128, 128)

    def test_001_basic_inference(self):
        pass
        # data = np.exp(1j*np.pi*np.linspace(0, 128*128*100 - 1,
        #               128*128*100, endpoint=True)/256).astype(np.complex64)

        # source = blocks.vector_source_c(data, False, 128*128)
        # infer_block = torch_script_infer_classify_vcvi(
        #     "vt_cnn2.pt", 0, 128, 128)
        # sink = blocks.vector_sink_i(128*128)

        # self.tb.connect(source, infer_block, sink)
        # self.tb.run()

        # print(sink.data()[:256])
        # expected = np.fft.fft(data[:256])
        # output = np.asarray(sink.data()[:256])

        # self.assertTrue(np.allclose(output, expected, atol=.1))


if __name__ == '__main__':
    gr_unittest.run(qa_torch_script_infer_classify_vcvi)
