#ifndef MSHADOW_EXTENSION_REC2ELL_FORWARD_H_
#define MSHADOW_EXTENSION_REC2ELL_FORWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename OffExp, typename RoiExp, typename DType, int srcdim>
struct Rec2EllfExp:
	public MakeTensorExp<Rec2EllfExp<OffExp, RoiExp, DType, srcdim>,
						 OffExp, srcdim, DType> {
	const OffExp &off_src_;
	const RoiExp &roi_src_;
	float spatial_scale_;

	Rec2EllfExp(const OffExp &off_src, const RoiExp &roi_src,
			   float spatial_scale)
		: off_src_(off_src), roi_src_(roi_src), spatial_scale_(spatial_scale) {
		Shape<srcdim> oshape = ShapeCheck<srcdim, OffExp>::Check(off_src_);
		this->shape_ = oshape;
	}
};

template<typename OffExp, typename RoiExp, typename DType, int etype>
inline Rec2EllfExp<OffExp, RoiExp, DType, ExpInfo<OffExp>::kDim>
rec2ell_forward(const Exp<OffExp, DType, etype> &off_src,
				const Exp<RoiExp, DType, etype> &roi_src,
				float spatial_scale) {
	TypeCheckPass<ExpInfo<OffExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<RoiExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return Rec2EllfExp<OffExp, RoiExp, DType, ExpInfo<OffExp>::kDim>
		(off_src.self(), roi_src.self(), spatial_scale);
}

template<typename OffExp, typename RoiExp, typename DType, int srcdim>
struct Plan<Rec2EllfExp<OffExp, RoiExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const Rec2EllfExp<OffExp, RoiExp, DType, srcdim> &e)
			: off_src_(MakePlan(e.off_src_)),
			  roi_src_(MakePlan(e.roi_src_)),
			  spatial_scale_(e.spatial_scale_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const DType x = roi_src_.Eval(i, 0);
			const DType y = roi_src_.Eval(i, 1);
			const DType h = roi_src_.Eval(i, 2);
			const DType w = roi_src_.Eval(i, 3);
			if(j == 0)
				return x * off_src_.Eval(i, j);
			else if(j == 1)
				return y * off_src_.Eval(i, j);
			else if(j == 2)
				return x + h / 2.0 + off_src_.Eval(i, j) * h;
			else if(j == 3)
				return y + w / 2.0 + off_src_.Eval(i, j) * w;
			else
				return off_src_.Eval(i, j);
		}

	private:
		Plan<OffExp, DType> off_src_;
		Plan<RoiExp, DType> roi_src_;
		const float spatial_scale_;
};
}	// namespace expr
}	// namespace mshadow
#endif
