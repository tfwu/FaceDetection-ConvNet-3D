#ifndef MSHADOW_EXTENSION_ROI_UNWARP_DATA_H_
#define MSHADOW_EXTENSION_ROI_UNWARP_DATA_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename RoisExp, typename DType, int srcdim>
struct ROIUnWarpDataExp:
	public MakeTensorExp<ROIUnWarpDataExp<DataExp, RoisExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
	const DataExp &data_src_;
	const RoisExp &rois_src_;
	const DataExp &grad_src_;
	Shape<2> warped_shape_;
	float spatial_scale_;

	index_t num_rois_;
	index_t src_channel_;
	index_t src_height_;
	index_t src_width_;

	ROIUnWarpDataExp(const DataExp &data_src, const RoisExp &rois_src, const DataExp &grad_src,
				  Shape<2> warped_shape, float spatial_scale)
		: data_src_(data_src), rois_src_(rois_src), grad_src_(grad_src),
		  warped_shape_(warped_shape), spatial_scale_(spatial_scale) {
		
		Shape<srcdim> sshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
		Shape<2> rshape = ShapeCheck<2, RoisExp>::Check(rois_src_);
		this->src_channel_ = sshape[srcdim - 3];
		this->src_height_ = sshape[srcdim - 2];
		this->src_width_  = sshape[srcdim - 1];
		this->num_rois_ = rshape[0];
		this->shape_ = sshape;
	}
};

template<typename DataExp, typename RoisExp, typename DType, int etype>
inline ROIUnWarpDataExp<DataExp, RoisExp, DType, ExpInfo<DataExp>::kDim>
roi_unwarp_data(const Exp<DataExp, DType, etype> &data_src,
				const Exp<RoisExp, DType, etype> &rois_src,
				const Exp<DataExp, DType, etype> &grad_src,
				Shape<2> warped_shape, float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<RoisExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return ROIUnWarpDataExp<DataExp, RoisExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), rois_src.self(), grad_src.self(),  warped_shape, spatial_scale);
}

template<typename DataExp, typename RoisExp, typename DType, int srcdim>
struct Plan<ROIUnWarpDataExp<DataExp, RoisExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const ROIUnWarpDataExp<DataExp, RoisExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  rois_src_(MakePlan(e.rois_src_)),
			  grad_src_(MakePlan(e.grad_src_)),
			  num_rois_(e.num_rois_),
			  warped_shape_(e.warped_shape_),
			  spatial_scale_(e.spatial_scale_),
			  src_channel_(e.src_channel_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t v = j;
			const index_t u = i % src_height_;
			const index_t c = (i / src_height_) % src_channel_;
			DType res = DType(0);
			for(index_t ind = 0; ind < num_rois_; ind++)
			{
				const index_t x = round(rois_src_.Eval(ind, 0) * spatial_scale_);
				const index_t y = round(rois_src_.Eval(ind, 1) * spatial_scale_);
				const index_t h = round(rois_src_.Eval(ind, 2) * spatial_scale_);
				const index_t w = round(rois_src_.Eval(ind, 3) * spatial_scale_);
				for(index_t u_ = 0; u_ < warped_shape_[0]; u_++)
					for(index_t v_ = 0; v_ < warped_shape_[1]; v_++)
					{
						DType tmpx = max(DType(0), 1 - std::fabs(x + u_ * h / (DType)warped_shape_[0] - u));
						DType tmpy = max(DType(0), 1 - std::fabs(y + v_ * w / (DType)warped_shape_[1] - v));
						res += grad_src_.Eval(ind * src_channel_ * warped_shape_[0] + \
											  c * warped_shape_[0] + u_, v_) * tmpx * tmpy;
					}
			}
			return res;
		}

	private:
		Plan<DataExp, DType> data_src_;
		Plan<RoisExp, DType> rois_src_;
		Plan<DataExp, DType> grad_src_;
		const index_t num_rois_;
		Shape<2> warped_shape_;
		const float spatial_scale_;
		const index_t src_channel_, src_height_, src_width_;
};
}	// namespace expr
}	// namespace mshadow
#endif
