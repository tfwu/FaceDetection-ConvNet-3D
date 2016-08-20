#ifndef MSHADOW_EXTENSION_CONFIGPOOL_FORWARD_H_
#define MSHADOW_EXTENSION_CONFIGPOOL_FORWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename GTExp, typename DType, int srcdim>
struct ConfigPoolfExp:
	public MakeTensorExp<ConfigPoolfExp<DataExp, GTExp, DType, srcdim>,
						 DataExp, 2, DType> {
	const DataExp &data_src_;
	const DataExp &keypoints_src_;
	const GTExp &groundtruth_src_;

	index_t num_keypoints_;
	index_t ksize_x_;
	index_t ksize_y_;
	index_t channel_size_;
	float spatial_scale_;
	
	index_t src_height_;
	index_t src_width_;
	
	ConfigPoolfExp(const DataExp &data_src, const DataExp &keypoints_src, 
				   const GTExp &groundtruth_src, index_t num_keypoints, index_t ksize_x, 
				   index_t ksize_y, float spatial_scale)
		: data_src_(data_src), keypoints_src_(keypoints_src), groundtruth_src_(groundtruth_src), 
		  num_keypoints_(num_keypoints), ksize_x_(ksize_x), ksize_y_(ksize_y), spatial_scale_(spatial_scale) {
		Shape<srcdim> dshape = ShapeCheck<srcdim, DataExp>::Check(data_src_);
		Shape<2> gtshape = ShapeCheck<2, GTExp>::Check(groundtruth_src_);
		this->src_height_ = dshape[srcdim - 2];
		this->src_width_ = dshape[srcdim - 1];
		this->channel_size_ = dshape[1];
		this->shape_ = gtshape;
		this->shape_[1] = num_keypoints_ * ksize_x_ * ksize_y_ * dshape[1];
	}
};

template<typename DataExp, typename GTExp, typename DType, int etype>
inline ConfigPoolfExp<DataExp, GTExp, DType, ExpInfo<DataExp>::kDim>
configpool_forward(const Exp<DataExp, DType, etype> &data_src,
				   const Exp<DataExp, DType, etype> &keypoints_src,
				   const Exp<GTExp, DType, etype> &groundtruth_src,
				   index_t num_keypoints, index_t ksize_x, index_t ksize_y,
				   float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<GTExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return ConfigPoolfExp<DataExp, GTExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), keypoints_src.self(), groundtruth_src.self(), 
		 num_keypoints, ksize_x, ksize_y, spatial_scale);
}

template<typename DataExp, typename GTExp, typename DType, int srcdim>
struct Plan<ConfigPoolfExp<DataExp, GTExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const ConfigPoolfExp<DataExp, GTExp, DType, srcdim> &e)
			: data_src_(MakePlan(e.data_src_)),
			  keypoints_src_(MakePlan(e.keypoints_src_)),
			  groundtruth_src_(MakePlan(e.groundtruth_src_)),
			  num_keypoints_(e.num_keypoints_),
			  ksize_x_(e.ksize_x_),
			  ksize_y_(e.ksize_y_),
			  channel_size_(e.channel_size_),
			  spatial_scale_(e.spatial_scale_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}
		
		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t px = round(groundtruth_src_.Eval(i, 0) * spatial_scale_);
			const index_t py = round(groundtruth_src_.Eval(i, 1) * spatial_scale_);
			const index_t c = j % channel_size_;
			const index_t ky = (j / channel_size_) % ksize_y_;
			const index_t kx = (j / channel_size_ / ksize_y_) % ksize_x_;
			const index_t nk = (j / channel_size_ / ksize_y_ / ksize_x_) % num_keypoints_;

			const index_t x = round(keypoints_src_.Eval(px * src_width_ + py, nk * 2) * spatial_scale_) \
							  - ksize_x_ / 2 + kx;
			const index_t y = round(keypoints_src_.Eval(px * src_width_ + py, nk * 2 + 1) * spatial_scale_) \
							  - ksize_y_ / 2 + ky;

			if(x >= src_height_ || y >= src_width_)
				return DType(0);
			return data_src_.Eval(c * src_height_ + x, y);
		}

	private:
		Plan<DataExp, DType> data_src_, keypoints_src_;
		Plan<GTExp, DType> groundtruth_src_;
		const index_t num_keypoints_, ksize_x_, ksize_y_, channel_size_;
		const float spatial_scale_;
		const index_t src_height_, src_width_;
};
}
}
#endif
