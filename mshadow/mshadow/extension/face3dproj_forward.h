#ifndef MSHADOW_EXTENSION_FACE3DPROJ_FORWARD_H_
#define MSHADOW_EXTENSION_FACE3DPROJ_FORWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename MFaceExp, typename DType, int srcdim>
struct Face3DProjfExp:
	public MakeTensorExp<Face3DProjfExp<DataExp, MFaceExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
	
	const DataExp &data_src_;
	const MFaceExp &mface_src_;
	index_t num_keypoints_;
	float spatial_scale_;

	index_t num_parameters_;
	index_t src_height_;
	index_t src_width_;

	Face3DProjfExp(const DataExp &data_src, const MFaceExp &mface_src,
				  index_t num_keypoints, float spatial_scale)
		: data_src_(data_src), mface_src_(mface_src), num_keypoints_(num_keypoints), spatial_scale_(spatial_scale) {

		Shape<srcdim> sshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
		this->num_parameters_ = sshape[srcdim - 3];
		this->src_height_ = sshape[srcdim - 2];
		this->src_width_ = sshape[srcdim - 1];
		this->shape_ = sshape;
		this->shape_[1] = sshape[2];
		this->shape_[2] = sshape[3];
		this->shape_[3] = num_keypoints_ * 2;
	}
};

template<typename DataExp, typename MFaceExp, typename DType, int etype>
inline Face3DProjfExp<DataExp, MFaceExp, DType, ExpInfo<DataExp>::kDim>
face3dproj_forward(const Exp<DataExp, DType, etype> &data_src,
				   const Exp<MFaceExp, DType, etype> &mface_src,
				   index_t num_keypoints, float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 3>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<MFaceExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return Face3DProjfExp<DataExp, MFaceExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), mface_src.self(), num_keypoints, spatial_scale);
}

template<typename DataExp, typename MFaceExp, typename DType, int srcdim>
struct Plan<Face3DProjfExp<DataExp, MFaceExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const Face3DProjfExp<DataExp, MFaceExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  mface_src_(MakePlan(e.mface_src_)),
			  num_keypoints_(e.num_keypoints_),
			  spatial_scale_(e.spatial_scale_),
			  num_parameters_(e.num_parameters_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t dim = j % 2;
			const index_t nk = j / 2;
			const index_t w = i % src_width_;
			const index_t h = (i / src_width_) % src_height_;
			const index_t n = i / src_width_ / src_height_;

			DType a0 = data_src_.Eval(n * num_parameters_ * src_height_ +
									  (0 + dim * 4) * src_height_ + h, w);
			DType a1 = data_src_.Eval(n * num_parameters_ * src_height_ + 
									  (1 + dim * 4) * src_height_ + h, w);
			DType a2 = data_src_.Eval(n * num_parameters_ * src_height_ +
									  (2 + dim * 4) * src_height_ + h, w);
			DType a3 = data_src_.Eval(n * num_parameters_ * src_height_ +
									  (3 + dim * 4) * src_height_ + h, w);

			DType x0 = mface_src_.Eval(nk, 0);
			DType x1 = mface_src_.Eval(nk, 1);
			DType x2 = mface_src_.Eval(nk, 2);

			return a0 * x0 + a1 * x1 + a2 * x2 + a3 + \
				   (dim == 0 ? static_cast<DType>(h) / spatial_scale_ :
							   static_cast<DType>(w) / spatial_scale_);
		}

	private:
		Plan<DataExp, DType> data_src_;
		Plan<MFaceExp, DType> mface_src_;
		const index_t num_keypoints_;
		const float spatial_scale_;
		const index_t num_parameters_, src_height_, src_width_;
};
}	// namespace expr
}	// namespace mshadow
#endif	// MSHADOW_EXTENSION_FACE3DPROJ_FORWARD_H_
