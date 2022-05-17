/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/io_signature.h>
#include "zcu216_source_impl.h"

namespace gr {
  namespace torchdsp {

    using output_type = gr_complex;
    zcu216_source::sptr
    zcu216_source::make()
    {
      return gnuradio::make_block_sptr<zcu216_source_impl>();
    }


    /*
     * The private constructor
     */
    zcu216_source_impl::zcu216_source_impl()
      : gr::sync_block("zcu216_source",
              gr::io_signature::make(0, 0, 0),
              gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
    {}

    /*
     * Our virtual destructor.
     */
    zcu216_source_impl::~zcu216_source_impl()
    {
    }

    int
    zcu216_source_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      auto out = static_cast<output_type*>(output_items[0]);

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace torchdsp */
} /* namespace gr */
