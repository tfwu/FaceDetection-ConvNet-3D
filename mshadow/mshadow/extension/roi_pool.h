#ifndef MSHADOW_EXTENSION_ROI_POOL_H_
#define MSHADOW_EXTENSION_ROI_POOL_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename RoisExp, typename DType, int srcdim>
struct ROIPoolingExp:
	public MakeTensorExp<ROIPoolingExp<DataExp, RoisExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
	const DataExp &data_src_;
	const RoisExp &rois_src_;
	index_t num_rois_;
	Shape<2> pooled_shape_;
	float spatial_scale_;

	index_t src_channel_;
	index_t src_height_;
	index_t src_width_;

	ROIPoolingExp(const DataExp &data_src, const RoisExp &rois_src,
				  index_t num_rois, Shape<2> pooled_shape, float spatial_scale)
		: data_src_(data_src), rois_src_(rois_src), num_rois_(num_rois),
		  pooled_shape_(pooled_shape), spatial_scale_(spatial_scale) {
		
		Shape<srcdim> sshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
		this->src_channel_ = sshape[srcdim - 3];
		this->src_height_ = sshape[srcdim - 2];
		this->src_width_  = sshape[srcdim - 1];
		this->shape_ = sshape;
		this->shape_[0] = sshape[0] * num_rois;
		this->shape_[srcdim - 2] = pooled_shape_[0];
		this->shape_[srcdim - 1] = pooled_shape_[1];
	}
};

template<typename DataExp, typename RoisExp, typename DType, int etype>
inline ROIPoolingExp<DataExp, RoisExp, DType, ExpInfo<DataExp>::kDim>
roipool(const Exp<DataExp, DType, etype> &data_src,
		const Exp<RoisExp, DType, etype> &rois_src,
		index_t num_rois, Shape<2> pooled_shape, float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<RoisExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return ROIPoolingExp<DataExp, RoisExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), rois_src.self(), num_rois, pooled_shape, spatial_scale);
}

template<typename DataExp, typename RoisExp, typename DType, int srcdim>
struct Plan<ROIPoolingExp<DataExp, RoisExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const ROIPoolingExp<DataExp, RoisExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  rois_src_(MakePlan(e.rois_src_)),
			  num_rois_(e.num_rois_),
			  pooled_shape_(e.pooled_shape_),
			  spatial_scale_(e.spatial_scale_),
			  src_channel_(e.src_channel_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t px = j;
			const index_t py = i % pooled_shape_[0];
			const index_t c = (i / pooled_shape_[0]) % src_channel_;
			const index_t r = (i / pooled_shape_[0] / src_channel_) % num_rois_;
			const index_t n = i / pooled_shape_[0] / src_channel_ / num_rois_;

			index_t roi_start_h = round(rois_src_.Eval(n, r * 4) * spatial_scale_);
			index_t roi_start_w = round(rois_src_.Eval(n, r * 4 + 1) * spatial_scale_);
			index_t roi_end_h = round(rois_src_.Eval(n, r * 4 + 2) * spatial_scale_);
			index_t roi_end_w = round(rois_src_.Eval(n, r * 4 + 3) * spatial_scale_);

			index_t roi_height = max(roi_end_h - roi_start_h + 1, (index_t)1);
			index_t roi_width = max(roi_end_w - roi_start_w + 1, (index_t)1);

			DType bin_size_h = static_cast<DType>(roi_height) 
							   / static_cast<DType>(pooled_shape_[0]);
			DType bin_size_w = static_cast<DType>(roi_width)
							   / static_cast<DType>(pooled_shape_[1]);

			index_t hstart = static_cast<index_t>(floor(static_cast<DType>(py) * bin_size_h));
			index_t wstart = static_cast<index_t>(floor(static_cast<DType>(px) * bin_size_w));
			index_t hend = static_cast<index_t>(ceil(static_cast<DType>((py + 1) * bin_size_h)));
			index_t wend = static_cast<index_t>(ceil(static_cast<DType>((px + 1) * bin_size_w)));

			hstart = min(max(hstart + roi_start_h, (index_t)0), src_height_);
			hend = min(max(hend + roi_start_h, (index_t)0), src_height_);
			wstart = min(max(wstart + roi_start_w, (index_t)0), src_width_);
			wend = min(max(wend + roi_start_w, (index_t)0), src_width_);
			bool is_empty = (hend <= hstart) || (wend <= wstart);

			DType res = is_empty ? static_cast<DType>(0) : -FLT_MAX;
			for(index_t h = hstart; h < hend; ++h) {
				for(index_t w = wstart; w < wend; ++w) {
					res = max(res, data_src_.Eval(n * src_channel_ * src_height_ +
												  c * src_height_ + h, w));
				}
			}

			return res;
		}

		private:
			Plan<DataExp, DType> data_src_;
			Plan<RoisExp, DType> rois_src_;
			const index_t num_rois_;
			Shape<2> pooled_shape_;
			const float spatial_scale_;
			const index_t src_channel_, src_height_, src_width_;
};
}	// namespace expr
}	// namespace mshadow
#endif	// MSHADOW_EXTENSION_ROI_POOL_H_
