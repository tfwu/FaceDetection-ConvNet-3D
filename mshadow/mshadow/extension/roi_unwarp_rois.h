#ifndef MSHADOW_EXTENSION_ROI_UNWARP_ROIS_H_
#define MSHADOW_EXTENSION_ROI_UNWARP_ROIS_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename RoisExp, typename DType, int srcdim>
struct ROIUnWarpRoisExp:
	public MakeTensorExp<ROIUnWarpRoisExp<DataExp, RoisExp, DType, srcdim>,
						 RoisExp, 2, DType> {
	const DataExp &data_src_;
	const RoisExp &rois_src_;
	const DataExp &grad_src_;
	Shape<2> warped_shape_;
	float spatial_scale_;

	index_t num_rois_;
	index_t src_channel_;
	index_t src_height_;
	index_t src_width_;

	ROIUnWarpRoisExp(const DataExp &data_src, const RoisExp &rois_src, const DataExp &grad_src,
				  Shape<2> warped_shape, float spatial_scale)
		: data_src_(data_src), rois_src_(rois_src), grad_src_(grad_src),
		  warped_shape_(warped_shape), spatial_scale_(spatial_scale) {
		
		Shape<srcdim> sshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
		Shape<2> rshape = ShapeCheck<2, RoisExp>::Check(rois_src_);
		this->src_channel_ = sshape[srcdim - 3];
		this->src_height_ = sshape[srcdim - 2];
		this->src_width_  = sshape[srcdim - 1];
		this->num_rois_ = rshape[0];
		this->shape_ = rshape;
	}
};

template<typename DataExp, typename RoisExp, typename DType, int etype>
inline ROIUnWarpRoisExp<DataExp, RoisExp, DType, ExpInfo<DataExp>::kDim>
roi_unwarp_rois(const Exp<DataExp, DType, etype> &data_src,
				const Exp<RoisExp, DType, etype> &rois_src,
				const Exp<DataExp, DType, etype> &grad_src,
				Shape<2> warped_shape, float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<RoisExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return ROIUnWarpRoisExp<DataExp, RoisExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), rois_src.self(), grad_src.self(),  warped_shape, spatial_scale);
}

template<typename DataExp, typename RoisExp, typename DType, int srcdim>
struct Plan<ROIUnWarpRoisExp<DataExp, RoisExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const ROIUnWarpRoisExp<DataExp, RoisExp, DType, srcdim> &e)
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
			DType res = DType(0);
			for(index_t c = 0; c < src_channel_; c++)
				for(index_t u_ = 0; u_ < warped_shape_[0]; u_++)
					for(index_t v_ = 0; v_ < warped_shape_[1]; v_++)
					{
						DType tmpx = rois_src_.Eval(i, 0) + u_ * rois_src_.Eval(i, 2) / (DType)warped_shape_[0];
						DType tmpy = rois_src_.Eval(i, 1) + v_ * rois_src_.Eval(i, 3) / (DType)warped_shape_[1];
						DType ux[2] = {tmpx, tmpx + 1};
						DType uy[2] = {tmpy, tmpy + 1};
						for(index_t ii = 0; ii < 2; ii++)
							for(index_t jj = 0; jj < 2; jj++)
								if(ux[ii] < src_height_ && ux[ii] >= 0 && uy[jj] < src_width_ && uy[jj] >= 0)
								{
									if(j == 0 || j == 2)
									{
										res += grad_src_.Eval(i * src_channel_ * warped_shape_[0] + \
															  c * warped_shape_[0] + u_, v_) * \
											   data_src_.Eval(c * src_height_ + (index_t)ux[ii], (index_t)uy[jj]) * \
											   (1.0 - fabs(uy[jj] - tmpy)) * \
											   (ux[ii] - tmpx < 0 ? -1 : 1) * \
											   (j == 0 ? 1.0 : u_ / (DType)warped_shape_[0]);
									}
									else if(j == 1 || j == 3)
									{
										res += grad_src_.Eval(i * src_channel_ * warped_shape_[0] + \
															  c * warped_shape_[0] + u_, v_) * \
											   data_src_.Eval(c * src_height_ + (index_t)ux[ii], (index_t)uy[jj]) * \
											   (1.0 - fabs(ux[ii] - tmpx)) * \
											   (uy[jj] - tmpy < 0 ? -1 : 1) * \
											   (j == 1 ? 1.0 : v_ / (DType)warped_shape_[1]);
									}
								}
					}
			return res / spatial_scale_;
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
