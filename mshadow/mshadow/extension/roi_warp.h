#ifndef MSHADOW_EXTENSION_ROI_WARP_H_
#define MSHADOW_EXTENSION_ROI_WARP_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename GTExp, typename DType, int srcdim>
struct ROIWarpExp:
	public MakeTensorExp<ROIWarpExp<DataExp, GTExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
	const DataExp &data_src_;
	const DataExp &rois_src_;
	const GTExp &gt_src_;
	Shape<2> warped_shape_;
	float spatial_scale_;

	index_t num_gt_;
	index_t src_channel_;
	index_t src_height_;
	index_t src_width_;

	ROIWarpExp(const DataExp &data_src, const DataExp &rois_src, const GTExp &gt_src,
			   Shape<2> warped_shape, float spatial_scale)
		: data_src_(data_src), rois_src_(rois_src), gt_src_(gt_src), 
		  warped_shape_(warped_shape), spatial_scale_(spatial_scale) {
		
		Shape<srcdim> sshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
		Shape<2> gtshape = ShapeCheck<2, GTExp>::Check(gt_src_);
		this->src_channel_ = sshape[srcdim - 3];
		this->src_height_ = sshape[srcdim - 2];
		this->src_width_  = sshape[srcdim - 1];
		this->num_gt_ = gtshape[0];
		this->shape_ = sshape;
		this->shape_[0] = gtshape[0];
		this->shape_[srcdim - 2] = warped_shape_[0];
		this->shape_[srcdim - 1] = warped_shape_[1];
	}
};

template<typename DataExp, typename GTExp, typename DType, int etype>
inline ROIWarpExp<DataExp, GTExp, DType, ExpInfo<DataExp>::kDim>
roi_warp(const Exp<DataExp, DType, etype> &data_src,
		 const Exp<DataExp, DType, etype> &rois_src,
		 const Exp<GTExp, DType, etype> &gt_src,
		 Shape<2> warped_shape, float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<GTExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return ROIWarpExp<DataExp, GTExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), rois_src.self(), gt_src.self(), warped_shape, spatial_scale);
}

template<typename DataExp, typename GTExp, typename DType, int srcdim>
struct Plan<ROIWarpExp<DataExp, GTExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const ROIWarpExp<DataExp, GTExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  rois_src_(MakePlan(e.rois_src_)),
			  gt_src_(MakePlan(e.gt_src_)),
			  num_gt_(e.num_gt_),
			  warped_shape_(e.warped_shape_),
			  spatial_scale_(e.spatial_scale_),
			  src_channel_(e.src_channel_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t v_ = j;
			const index_t u_ = i % warped_shape_[0];
			const index_t c = (i / warped_shape_[0]) % src_channel_;
			const index_t ind = i / warped_shape_[0] / src_channel_;

			const index_t gt_x = round(gt_src_.Eval(ind, 0) * spatial_scale_);
			const index_t gt_y = round(gt_src_.Eval(ind, 1) * spatial_scale_);

			const index_t x = round(rois_src_.Eval(gt_x * src_width_ + gt_y, 0) * spatial_scale_);
			const index_t y = round(rois_src_.Eval(gt_x * src_width_ + gt_y, 1) * spatial_scale_);
			const index_t h = round(rois_src_.Eval(gt_x * src_width_ + gt_y, 2) * spatial_scale_);
			const index_t w = round(rois_src_.Eval(gt_x * src_width_ + gt_y, 3) * spatial_scale_);

			const DType tmpx = x + u_ * h / (DType)warped_shape_[0];
			const DType tmpy = y + v_ * w / (DType)warped_shape_[1];
			const DType ux[2] = { tmpx, tmpx + 1};
			const DType uy[2] = { tmpy, tmpy + 1};
			DType res = DType(0);
			for(index_t ii = 0; ii < 2; ii++)
				for(index_t jj = 0; jj < 2; jj++)
					if(ux[ii] < src_height_ && ux[ii] >= 0 && uy[jj] < src_width_ && uy[jj] >= 0)
						res += (1.0 - fabs((index_t)ux[ii] - tmpx)) * (1.0 - fabs((index_t)uy[jj] - tmpy)) * \
							   data_src_.Eval(c * src_height_ + (index_t)ux[ii], (index_t)uy[jj]);
			return res;
		}

	private:
		Plan<DataExp, DType> data_src_, rois_src_;
		Plan<GTExp, DType> gt_src_;
		const index_t num_gt_;
		Shape<2> warped_shape_;
		const float spatial_scale_;
		const index_t src_channel_, src_height_, src_width_;
};
}	// namespace expr
}	// namespace mshadow
#endif
