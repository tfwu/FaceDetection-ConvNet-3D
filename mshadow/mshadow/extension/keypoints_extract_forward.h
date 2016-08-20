#ifndef MSHADOW_EXTENSION_KEYPOINTS_EXTRACT_FORWARD_H_
#define MSHADOW_EXTENSION_KEYPOINTS_EXTRACT_FORWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename GTExp, typename DataExp, typename DType, int srcdim>
struct KeypointsExtractfExp:
	public MakeTensorExp<KeypointsExtractfExp<GTExp, DataExp, DType, srcdim>,
						 GTExp, srcdim, DType> {
	
	const GTExp &groundtruth_src_;
	const DataExp &data_src_;
	index_t anchor_;
	index_t num_keypoints_;
	float spatial_scale_;
	
	index_t src_height_;
	index_t src_width_;

	KeypointsExtractfExp(const GTExp &groundtruth_src, const DataExp &data_src, index_t anchor, index_t num_keypoints, float spatial_scale)
		: groundtruth_src_(groundtruth_src), data_src_(data_src), anchor_(anchor), num_keypoints_(num_keypoints), spatial_scale_(spatial_scale) {
			Shape<srcdim> gtshape = ShapeCheck<srcdim, GTExp>::Check(groundtruth_src_);
			Shape<4> dshape = ShapeCheck<4, DataExp>::Check(data_src_);
			this->src_height_ = dshape[1];
			this->src_width_ = dshape[2];
			this->shape_ = gtshape;
			this->shape_[1] = num_keypoints_ * 2;
		}
};

template<typename GTExp, typename DataExp, typename DType, int etype>
inline KeypointsExtractfExp<GTExp, DataExp, DType, ExpInfo<GTExp>::kDim>
keypoints_extract_forward(const Exp<GTExp, DType, etype> &groundtruth_src,
						  const Exp<DataExp, DType, etype> &data_src,
						  index_t anchor, index_t num_keypoints,
						  float spatial_scale) {
	TypeCheckPass<ExpInfo<GTExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return KeypointsExtractfExp<GTExp, DataExp, DType, ExpInfo<GTExp>::kDim>
		(groundtruth_src.self(), data_src.self(), anchor, num_keypoints, 
		 spatial_scale);
}

template<typename GTExp, typename DataExp, typename DType, int srcdim>
struct Plan<KeypointsExtractfExp<GTExp, DataExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const KeypointsExtractfExp<GTExp, DataExp, DType, srcdim> &e)
			: groundtruth_src_(MakePlan(e.groundtruth_src_)),
			  data_src_(MakePlan(e.data_src_)),
			  anchor_(e.anchor_),
			  num_keypoints_(e.num_keypoints_),
			  spatial_scale_(e.spatial_scale_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}
		
		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t d = j % 2;
			const index_t px = round(groundtruth_src_.Eval(i, 0) * spatial_scale_);
			const index_t py = round(groundtruth_src_.Eval(i, 1) * spatial_scale_);
			return data_src_.Eval(px * src_width_ + py, anchor_ * 2 + d);
		}

	private:
		Plan<GTExp, DType> groundtruth_src_;
		Plan<DataExp, DType> data_src_;
		const index_t anchor_, num_keypoints_;
		const float spatial_scale_;
		const index_t src_height_, src_width_;
};
}
}
#endif


