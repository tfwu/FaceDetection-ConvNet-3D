#ifndef MSHADOW_EXTENSION_BOX_PREDICT_FORWARD_H_
#define MSHADOW_EXTENSION_BOX_PREDICT_FORWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename DType, int srcdim>
struct BoxPredictfExp:
	public MakeTensorExp<BoxPredictfExp<DataExp, DType, srcdim>,
						 DataExp, srcdim, DType> {

		const DataExp &data_src_;
		index_t num_keypoints_;
		index_t src_height_;
		index_t src_width_;

		BoxPredictfExp(const DataExp &data_src)
			: data_src_(data_src) {
			Shape<srcdim> dshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
			this->num_keypoints_ = dshape[3] / 2;
			this->src_height_ = dshape[1];
			this->src_width_ = dshape[2];
			this->shape_ = dshape;
			this->shape_[3] = 4;
		}
};

template<typename DataExp, typename DType, int etype>
inline BoxPredictfExp<DataExp, DType, ExpInfo<DataExp>::kDim>
box_predict_forward(const Exp<DataExp, DType, etype> &data_src) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return BoxPredictfExp<DataExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self());
}

template<typename DataExp, typename DType, int srcdim>
struct Plan<BoxPredictfExp<DataExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const BoxPredictfExp<DataExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  num_keypoints_(e.num_keypoints_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t y = i % src_width_;
			const index_t x = (i / src_width_) % src_height_;
			DType min_x = FLT_MAX, min_y = FLT_MAX;
			DType max_x = -FLT_MAX, max_y = -FLT_MAX;

			for(index_t k = 0; k < num_keypoints_; k++)
			{
				DType xx = data_src_.Eval(x * src_width_ + y, k * 2);
				DType yy = data_src_.Eval(x * src_width_ + y, k * 2 + 1);
				min_x = min(min_x, xx), min_y = min(min_y, yy);
				max_x = max(max_x, xx), max_y = max(max_y, yy);
			}

			if(j == 0)
				return round(min_x);
			else if(j == 1)
				return round(min_y);
			else if(j == 2)
				return round(max_x) - round(min_x);
			else if(j == 3)
				return round(max_y) - round(min_y);
			return DType(0);
		}

	private:
		Plan<DataExp, DType> data_src_;
		const index_t num_keypoints_;
		const index_t src_height_, src_width_;
};
}	// namespace expr
}	// namespace mshadow
#endif	
