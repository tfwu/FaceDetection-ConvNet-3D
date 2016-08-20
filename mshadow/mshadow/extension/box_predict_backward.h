#ifndef MSHADOW_EXTENSION_BOX_PREDICT_BACKWARD_H_
#define MSHADOW_EXTENSION_BOX_PREDICT_BACKWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename DType, int srcdim>
struct BoxPredictbExp:
	public MakeTensorExp<BoxPredictbExp<DataExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
		const DataExp &data_src_;
		const DataExp &out_;
		const DataExp &grad_;

		index_t num_keypoints_;
		index_t src_height_;
		index_t src_width_;

		BoxPredictbExp(const DataExp &data_src, const DataExp &out, const DataExp &grad)
			: data_src_(data_src), out_(out), grad_(grad) {
			Shape<srcdim> sshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
			this->num_keypoints_ = sshape[3] / 2;
			this->src_height_ = sshape[1];
			this->src_width_ = sshape[2];
			this->shape_ = sshape;
	}
};

template<typename DataExp, typename DType, int etype>
inline BoxPredictbExp<DataExp, DType, ExpInfo<DataExp>::kDim>
box_predict_backward(const Exp<DataExp, DType, etype> &data_src,
					 const Exp<DataExp, DType, etype> &out,
					 const Exp<DataExp, DType, etype> &grad) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return BoxPredictbExp<DataExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), out.self(), grad.self());
}

template<typename DataExp, typename DType, int srcdim>
struct Plan<BoxPredictbExp<DataExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const BoxPredictbExp<DataExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  out_(MakePlan(e.out_)),
			  grad_(MakePlan(e.grad_)),
			  num_keypoints_(e.num_keypoints_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t ind = j % 2;
			DType res = DType(0);
			
			const index_t d = round(data_src_.Eval(i, j));
			const index_t base = round(out_.Eval(i, ind));
			const index_t offs = round(out_.Eval(i, ind + 2));
			if(d == base)
				return grad_.Eval(i, ind) -  grad_.Eval(i, ind + 2);
			else if(d == base + offs)
				return grad_.Eval(i, ind + 2);

			return res;
		}

	private:
		Plan<DataExp, DType> data_src_, out_, grad_;
		const index_t num_keypoints_;
		const index_t src_height_, src_width_;
};
}
}
#endif
