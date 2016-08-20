#ifndef MSHADOW_EXTENSION_GEN_ELL_LABEL_FORWARD_H_
#define MSHADOW_EXTENSION_GEN_ELL_LABEL_FORWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename RoiExp, typename LabelExp, typename DType, int srcdim>
struct GenEllLabelfExp:
	public MakeTensorExp<GenEllLabelfExp<RoiExp, LabelExp, DType, srcdim>,
						 LabelExp, 2, DType> {
	const RoiExp &roi_src_;
	const LabelExp &label_src_;
	const LabelExp &gt_src_;
	float spatial_scale_;

	index_t src_height_;
	index_t src_width_;

	GenEllLabelfExp(const RoiExp &roi_src, const LabelExp &label_src, const LabelExp &gt_src, float spatial_scale)
		: roi_src_(roi_src), label_src_(label_src), gt_src_(gt_src), 
		  spatial_scale_(spatial_scale) {
		Shape<srcdim> rshape = ShapeCheck<srcdim, RoiExp>::Check(roi_src_);
		Shape<2> oshape = ShapeCheck<2, LabelExp>::Check(label_src_);
		src_height_ = rshape[1];
		src_width_ = rshape[2];
		this->shape_ = oshape;
	}
};

template<typename RoiExp, typename LabelExp, typename DType, int etype>
inline GenEllLabelfExp<RoiExp, LabelExp, DType, ExpInfo<RoiExp>::kDim>
gen_ell_label_forward(const Exp<RoiExp, DType, etype> &roi_src,
					  const Exp<LabelExp, DType, etype> &label_src,
					  const Exp<LabelExp, DType, etype> &gt_src,
					  const float spatial_scale) {
	TypeCheckPass<ExpInfo<RoiExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<LabelExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return GenEllLabelfExp<RoiExp, LabelExp, DType, ExpInfo<RoiExp>::kDim>
		(roi_src.self(), label_src.self(), gt_src.self(), spatial_scale);
}

template<typename RoiExp, typename LabelExp, typename DType, int srcdim>
struct Plan<GenEllLabelfExp<RoiExp, LabelExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const GenEllLabelfExp<RoiExp, LabelExp, DType, srcdim> &e)
			: roi_src_(MakePlan(e.roi_src_)),
			  label_src_(MakePlan(e.label_src_)),
			  gt_src_(MakePlan(e.gt_src_)),
			  spatial_scale_(e.spatial_scale_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t gt_x = round(gt_src_.Eval(i, 0) * spatial_scale_);
			const index_t gt_y = round(gt_src_.Eval(i, 1) * spatial_scale_);
			const DType x = roi_src_.Eval(gt_x * src_width_ + gt_y, 0);
			const DType y = roi_src_.Eval(gt_x * src_width_ + gt_y, 1);
			const DType h = roi_src_.Eval(gt_x * src_width_ + gt_y, 2);
			const DType w = roi_src_.Eval(gt_x * src_width_ + gt_y, 3);
			if(j == 0)
				return label_src_.Eval(i, 0) / h;
			else if(j == 1)
				return label_src_.Eval(i, 1) / w;
			else if(j == 2)
				return (label_src_.Eval(i, 2) - x) / h;
			else if(j == 3)
				return (label_src_.Eval(i, 3) - y) / w;
			else
				return label_src_.Eval(i, 4);
		}

	private:
		Plan<RoiExp, DType> roi_src_;
		Plan<LabelExp, DType> label_src_, gt_src_;
		const float spatial_scale_;
		const index_t src_height_, src_width_;

};
}	// namespace expr
}	// namespace mshadow
#endif
