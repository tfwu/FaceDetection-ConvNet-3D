#ifndef MSHADOW_FACE3DPROJ_BACKWARD_H_
#define MSHADOW_FACE3DPROJ_BACKWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename MFaceExp, typename DType, int srcdim>
struct Face3DProjbExp:
	public MakeTensorExp<Face3DProjbExp<DataExp, MFaceExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
		const DataExp &data_src_;
		const MFaceExp &mface_src_;
		const DataExp &grad_;
		index_t num_keypoints_;

		index_t num_parameters_;
		index_t src_height_;
		index_t src_width_;

		Face3DProjbExp(const DataExp &data_src, const MFaceExp &mface_src,
					   const DataExp &grad, index_t num_keypoints)
			: data_src_(data_src), mface_src_(mface_src), grad_(grad), num_keypoints_(num_keypoints) {
			Shape<srcdim> sshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
			
			this->num_parameters_ = sshape[srcdim - 3];
			this->src_height_ = sshape[srcdim - 2];
			this->src_width_ = sshape[srcdim - 1];
			this->shape_ = sshape;
	}
};

template<typename DataExp, typename MFaceExp, typename DType, int etype>
inline Face3DProjbExp<DataExp, MFaceExp, DType, ExpInfo<DataExp>::kDim>
face3dproj_backward(const Exp<DataExp, DType, etype> &data_src,
					const Exp<MFaceExp, DType, etype> &m_face,
					const Exp<DataExp, DType, etype> &grad,
					index_t num_keypoints) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 3>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<MFaceExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return Face3DProjbExp<DataExp, MFaceExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), m_face.self(), grad.self(), num_keypoints);
}

template<typename DataExp, typename MFaceExp, typename DType, int srcdim>
struct Plan<Face3DProjbExp<DataExp, MFaceExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const Face3DProjbExp<DataExp, MFaceExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  mface_src_(MakePlan(e.mface_src_)),
			  grad_(MakePlan(e.grad_)),
			  num_keypoints_(e.num_keypoints_),
			  num_parameters_(e.num_parameters_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;

			const index_t w = j;
			const index_t h = i % src_height_;
			const index_t np = (i / src_height_) % num_parameters_;
			const index_t n = i / src_height_ / num_parameters_;

			const index_t dim = np / 4;
			const index_t ind = np % 4;

			DType res = static_cast<DType>(0);
			for(index_t k = 0; k < num_keypoints_; k++) {
				DType grad = grad_.Eval(n * src_height_ * src_width_ + 
										h * src_width_ + w, k * 2 + dim);
				DType x = ind == 3 ? 1 : mface_src_.Eval(k, ind);
				res += grad * x;
			}

			return res;
		}

	private:
		Plan<DataExp, DType> data_src_;
		Plan<MFaceExp, DType> mface_src_;
		Plan<DataExp, DType> grad_;
		const index_t num_keypoints_, num_parameters_;
		const index_t src_height_, src_width_;
};
}
}
#endif
