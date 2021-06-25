/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_fir_filter_ccc_H
#define INCLUDED_TORCHDSP_fir_filter_ccc_H

#include <gnuradio/sync_decimator.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API fir_filter_ccc : virtual public gr::sync_decimator {
public:
  typedef std::shared_ptr<fir_filter_ccc> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of torchdsp::fir_filter_ccc.
   *
   * To avoid accidental use of raw pointers, torchdsp::fir_filter_ccc's
   * constructor is in a private implementation
   * class. torchdsp::fir_filter_ccc::make is the public interface for
   * creating new instances.
   */
  static sptr make(const std::vector<gr_complex> &taps,
                   unsigned int downsample_rate, unsigned int device_num);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_fir_filter_ccc_H */
