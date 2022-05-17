/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_ZCU216_SOURCE_IMPL_H
#define INCLUDED_TORCHDSP_ZCU216_SOURCE_IMPL_H

#include <torchdsp/zcu216_source.h>

namespace gr {
  namespace torchdsp {

    class zcu216_source_impl : public zcu216_source
    {
     private:
      // Nothing to declare in this block.

     public:
      zcu216_source_impl();
      ~zcu216_source_impl();

      // Where all the action really happens
      int work(
              int noutput_items,
              gr_vector_const_void_star &input_items,
              gr_vector_void_star &output_items
      );
    };

  } // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_ZCU216_SOURCE_IMPL_H */
