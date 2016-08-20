#ifndef MSHADOW_EXTENSION_ROI_UNPOOL_H_
#define MSHADOW_EXTENSION_ROI_UNPOOL_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename RoisExp, typename DType, int srcdim>
struct ROIUnPoolingExp:
	public MakeTensorExp<ROIUnPoolingExp<DataExp, RoisExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
	const DataExp &data_src_;
	const RoisExp &rois_src_;
	const DataExp &data_pooled_;
	const DataExp &grad_pooled_;
	index_t num_rois_;
	Shape<2> pooled_shape_;
	float spatial_scale_;

	index_t src_channel_;
	index_t src_height_;
	index_t src_width_;

	ROIUnPoolingExp(const DataExp &data_src, const RoisExp &rois_src, 
					const DataExp &data_pooled, const DataExp &grad_pooled,
					index_t num_rois, Shape<2> pooled_shape, float spatial_scale)
		: data_src_(data_src), rois_src_(rois_src), data_pooled_(data_pooled),
		  grad_pooled_(grad_pooled), num_rois_(num_rois), pooled_shape_(pooled_shape),
		  spatial_scale_(spatial_scale) {

		Shape<srcdim> sshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
		this->src_channel_ = sshape[srcdim - 3];
		this->src_height_ = sshape[srcdim - 2];
		this->src_width_ = sshape[srcdim - 1];
		this->shape_ = sshape;
	}
};

template<typename DataExp, typename RoisExp, typename DType, int etype>
inline ROIUnPoolingExp<DataExp, RoisExp, DType, ExpInfo<DataExp>::kDim>
roiunpool(const Exp<DataExp, DType, etype> &data_src,
		  const Exp<RoisExp, DType, etype> &rois_src,
		  const Exp<DataExp, DType, etype> &data_pooled,
		  const Exp<DataExp, DType, etype> &grad_pooled,
		  index_t num_rois, Shape<2> pooled_shape, float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<RoisExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return ROIUnPoolingExp<DataExp, RoisExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), rois_src.self(), data_pooled.self(), grad_pooled.self(), num_rois, pooled_shape, spatial_scale);
}

template<typename DataExp, typename RoisExp, typename DType, int srcdim>
struct Plan<ROIUnPoolingExp<DataExp, RoisExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const ROIUnPoolingExp<DataExp, RoisExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  rois_src_(MakePlan(e.rois_src_)),
			  data_pooled_(MakePlan(e.data_pooled_)),
			  grad_pooled_(MakePlan(e.grad_pooled_)),
			  num_rois_(e.num_rois_),
			  pooled_shape_(e.pooled_shape_),
			  spatial_scale_(e.spatial_scale_),
			  src_channel_(e.src_channel_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const DType vsrc = data_src_.Eval(i, j);
			index_t w = j;
			index_t h = i % src_height_;
			index_t c = (i / src_height_) % src_channel_;
			index_t n = i / src_height_ / src_channel_;

			DType res = static_cast<DType>(0);
			for(index_t r = 0; r < num_rois_; ++r) {

				index_t roi_start_h = round(rois_src_.Eval(n, r * 4) 
											* spatial_scale_);
				index_t roi_start_w = round(rois_src_.Eval(n, r * 4 + 1)
											* spatial_scale_);
				index_t roi_end_h = round(rois_src_.Eval(n, r * 4 + 2) 
										  * spatial_scale_);
				index_t roi_end_w = round(rois_src_.Eval(n, r * 4 + 3) 
										  * spatial_scale_);

				const bool in_roi = (h >= roi_start_h && h <= roi_end_h &&
									 w >= roi_start_w && w <= roi_end_w);
				if(!in_roi) {
					continue;
				}

				index_t roi_height = max(roi_end_h - roi_start_h + 1, (index_t)1);
				index_t roi_width = max(roi_end_w - roi_start_w + 1, (index_t)1);

				DType bin_size_h = static_cast<DType>(roi_height)
								   / static_cast<DType>(pooled_shape_[0]);
				DType bin_size_w = static_cast<DType>(roi_width)
								   / static_cast<DType>(pooled_shape_[1]);

				index_t phstart = floor(static_cast<DType>(h - roi_start_h) / bin_size_h);
				index_t phend = ceil(static_cast<DType>(h - roi_start_h + 1) / bin_size_h);
				index_t pwstart = floor(static_cast<DType>(w - roi_start_w) / bin_size_w);
				index_t pwend = ceil(static_cast<DType>(w - roi_start_w + 1) / bin_size_w);

				phstart = min(max(phstart, (index_t)0), pooled_shape_[0]);
				phend = min(max(phend, (index_t)0), pooled_shape_[0]);
				pwstart = min(max(pwstart, (index_t)0), pooled_shape_[1]);
				pwend = min(max(pwend, (index_t)0), pooled_shape_[1]);
				
				for(index_t ph = phstart; ph < phend; ++ph) {
					for(index_t pw = pwstart; pw < pwend; ++pw) {

						index_t indh = n * num_rois_ * src_channel_ * pooled_shape_[0] + 
									   r * src_channel_ * pooled_shape_[0] +
									   c * pooled_shape_[0] + ph;

						if(vsrc == data_pooled_.Eval(indh, pw))
							res += grad_pooled_.Eval(indh, pw);
					}
				}
			}

			return res;
		}

		private:
			Plan<DataExp, DType> data_src_;
			Plan<RoisExp, DType> rois_src_;
			Plan<DataExp, DType> data_pooled_, grad_pooled_;
			const index_t num_rois_;
			Shape<2> pooled_shape_;
			const float spatial_scale_;
			const index_t src_channel_, src_height_, src_width_;
};
}	// namespace expr
}	// namespace mshadow
#endif	// MSHADOW_EXTENSION_ROI_UNPOOL_H_





