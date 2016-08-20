#ifndef MSHADOW_EXTENSION_FACE_ELEMENT_SUM_FORWARD_H_
#define MSHADOW_EXTENSION_FACE_ELEMENT_SUM_FORWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename KeyExp, typename DType, int srcdim>
struct FaceElementSumfExp:
	public MakeTensorExp<FaceElementSumfExp<DataExp, KeyExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
	const DataExp &data_src_;
	const KeyExp &keypoints_src_;
	const DataExp &groundtruth_src_;

	index_t num_keypoints_;
	float spatial_scale_;

	index_t src_height_;
	index_t src_width_;

	FaceElementSumfExp(const DataExp &data_src, const KeyExp &keypoints_src, const DataExp &groundtruth_src, index_t num_keypoints, float spatial_scale)
		: data_src_(data_src), keypoints_src_(keypoints_src), groundtruth_src_(groundtruth_src), num_keypoints_(num_keypoints), spatial_scale_(spatial_scale) {
			Shape<srcdim> dshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
			Shape<4> kshape = ShapeCheck<4, KeyExp>::Check(keypoints_src_);
			this->src_height_ = kshape[1];
			this->src_width_ = kshape[2];
			this->shape_ = dshape;
		}
};

template<typename DataExp, typename KeyExp, typename DType, int etype>
inline FaceElementSumfExp<DataExp, KeyExp, DType, ExpInfo<DataExp>::kDim>
face_element_sum_forward(const Exp<DataExp, DType, etype> &data_src,
						 const Exp<KeyExp, DType, etype> &keypoints_src,
						 const Exp<DataExp, DType, etype> &groundtruth_src,
						 index_t num_keypoints, float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<KeyExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return FaceElementSumfExp<DataExp, KeyExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), keypoints_src.self(), groundtruth_src.self(),
		 num_keypoints, spatial_scale);
}

template<typename DataExp, typename KeyExp, typename DType, int srcdim>
struct Plan<FaceElementSumfExp<DataExp, KeyExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const FaceElementSumfExp<DataExp, KeyExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  keypoints_src_(MakePlan(e.keypoints_src_)),
			  groundtruth_src_(MakePlan(e.groundtruth_src_)),
			  num_keypoints_(e.num_keypoints_),
			  spatial_scale_(e.spatial_scale_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}
		
		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t px = round(groundtruth_src_.Eval(i, 0) * spatial_scale_);
			const index_t py = round(groundtruth_src_.Eval(i, 1) * spatial_scale_);
			return data_src_.Eval(i, j) + \
				   keypoints_src_.Eval(px * src_width_ + py, j);
		}

	private:
		Plan<DataExp, DType> data_src_;
		Plan<KeyExp, DType> keypoints_src_;
		Plan<DataExp, DType> groundtruth_src_;
		const index_t num_keypoints_;
		const float spatial_scale_;
		const index_t src_height_, src_width_;
};
}
}
#endif

