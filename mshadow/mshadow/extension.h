/*!
 * Copyright by Contributors
 * \file extension.h
 * \brief some extension of expressions,
 *  used to support something beyond elementwise op
 * \author Tianqi Chen, Bing Xu
 */
#ifndef MSHADOW_EXTENSION_H_
#define MSHADOW_EXTENSION_H_
#include "./expr_engine-inl.h"
#include "./extension/broadcast.h"
#include "./extension/unpack_patch2col.h"
#include "./extension/pack_col2patch.h"
#include "./extension/reshape.h"
#include "./extension/swapaxis.h"
#include "./extension/reduceto1d.h"
#include "./extension/spatial_pool.h"
#include "./extension/spatial_unpool.h"
#include "./extension/channel_pool.h"
#include "./extension/channel_unpool.h"
#include "./extension/pad.h"
#include "./extension/crop.h"
#include "./extension/mirror.h"
#include "./extension/concat.h"
#include "./extension/implicit_gemm.h"
#include "./extension/choose.h"
#include "./extension/fill.h"
#include "./extension/one_hot.h"
#include "./extension/slice.h"
#include "./extension/take.h"
#include "./extension/take_grad.h"
#include "./extension/reduce_with_axis.h"
#include "./extension/broadcast_with_axis.h"
#include "./extension/spatial_upsampling_nearest.h"
#include "./extension/roi_pool.h"
#include "./extension/roi_unpool.h"
#include "./extension/face3dproj_forward.h"
#include "./extension/face3dproj_backward.h"
#include "./extension/configpool_forward.h"
#include "./extension/configpool_backward.h"
#include "./extension/face_element_sum_forward.h"
#include "./extension/keypoints_extract_forward.h"
#include "./extension/box_predict_forward.h"
#include "./extension/box_predict_backward.h"
#include "./extension/roi_warp.h"
//#include "./extension/roi_unwarp_data.h"
//#include "./extension/roi_unwarp_rois.h"
#include "./extension/rec2ell_forward.h"
#include "./extension/rec2ell_backward.h"
#include "./extension/gen_ell_label.h"
#endif  // MSHADOW_EXTENSION_H_
