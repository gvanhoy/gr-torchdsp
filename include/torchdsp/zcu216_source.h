/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_ZCU216_SOURCE_H
#define INCLUDED_TORCHDSP_ZCU216_SOURCE_H

#include <torchdsp/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace torchdsp {

    /*!
     * \brief <+description of block+>
     * \ingroup torchdsp
     *
     */
    class TORCHDSP_API zcu216_source : virtual public gr::sync_block
    {
     public:
      typedef std::shared_ptr<zcu216_source> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of torchdsp::zcu216_source.
       *
       * To avoid accidental use of raw pointers, torchdsp::zcu216_source's
       * constructor is in a private implementation
       * class. torchdsp::zcu216_source::make is the public interface for
       * creating new instances.
       */
      static sptr make();
    };

  } // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_ZCU216_SOURCE_H */
