/* -*- c++ -*- */
/*
 * Copyright 2021 gr-torchdsp author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_TORCH_SCRIPT_INFER_CLASSIFY_VCVI_H
#define INCLUDED_TORCHDSP_TORCH_SCRIPT_INFER_CLASSIFY_VCVI_H

#include <gnuradio/sync_block.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API torch_script_infer_classify_vcvi
    : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<torch_script_infer_classify_vcvi> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of
   * torchdsp::torch_script_infer_classify_vcvi.
   *
   * To avoid accidental use of raw pointers,
   * torchdsp::torch_script_infer_classify_vcvi's constructor is in a private
   * implementation class. torchdsp::torch_script_infer_classify_vcvi::make is
   * the public interface for creating new instances.
   */
  static sptr make(std::string jit_model_path, unsigned int device_num,
                   unsigned int batch_size, unsigned int num_samples);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_TORCH_SCRIPT_INFER_CLASSIFY_VCVI_H */
