#ifndef MSHADOW_EXTENSION_CONFIGPOOL_BACKWARD_H_
#define MSHADOW_EXTENSION_CONFIGPOOL_BACKWARD_H_
#include <algorithm>
#include <cmath>
#include "../extension.h"
namespace mshadow {
namespace expr {

template<typename DataExp, typename GTExp, typename DType, int srcdim>
struct ConfigPoolbExp:
	public MakeTensorExp<ConfigPoolbExp<DataExp, GTExp, DType, srcdim>,
						 DataExp, srcdim, DType> {
	const DataExp &keypoints_src_;
	const GTExp &groundtruth_src_;
	const GTExp &grad_;

	index_t num_keypoints_;
	index_t num_groundtruth_;
	index_t ksize_x_;
	index_t ksize_y_;
	index_t channel_size_;
	float spatial_scale_;

	index_t src_height_;
	index_t src_width_;

	ConfigPoolbExp(const DataExp &data_src, const DataExp &keypoints_src, const GTExp &groundtruth_src, const GTExp &grad, index_t num_keypoints, index_t ksize_x, index_t ksize_y, float spatial_scale)
		: keypoints_src_(keypoints_src), groundtruth_src_(groundtruth_src), grad_(grad), num_keypoints_(num_keypoints), ksize_x_(ksize_x), ksize_y_(ksize_y), spatial_scale_(spatial_scale) {
		Shape<srcdim> dshape = ShapeCheck<srcdim, DataExp>::Check(data_src);
		Shape<2> gtshape = ShapeCheck<2, GTExp>::Check(groundtruth_src_);
		
		this->src_height_ = dshape[srcdim - 2];
		this->src_width_ = dshape[srcdim - 1];
		this->num_groundtruth_ = gtshape[0];
		this->channel_size_ = dshape[1];
		this->shape_ = dshape;
	}
};

template<typename DataExp, typename GTExp, typename DType, int etype>
inline ConfigPoolbExp<DataExp, GTExp, DType, ExpInfo<DataExp>::kDim>
configpool_backward(const Exp<DataExp, DType, etype> &data_src,
				    const Exp<DataExp, DType, etype> &keypoints_src,
				    const Exp<GTExp, DType, etype> &groundtruth_src,
					const Exp<GTExp, DType, etype> &grad,
				    index_t num_keypoints, index_t ksize_x, index_t ksize_y,
				    float spatial_scale) {
	TypeCheckPass<ExpInfo<DataExp>::kDim >= 4>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	TypeCheckPass<ExpInfo<GTExp>::kDim >= 2>
		::Error_Expression_Does_Not_Meet_Dimension_Req();
	return ConfigPoolbExp<DataExp, GTExp, DType, ExpInfo<DataExp>::kDim>
		(data_src.self(), keypoints_src.self(), groundtruth_src.self(),
		 grad.self(), num_keypoints, ksize_x, ksize_y, spatial_scale);
}

template<typename DataExp, typename GTExp, typename DType, int srcdim>
struct Plan<ConfigPoolbExp<DataExp, GTExp, DType, srcdim>, DType>
{
	public:
		explicit Plan(const ConfigPoolbExp<DataExp, GTExp, DType, srcdim> &e)
			: keypoints_src_(MakePlan(e.keypoints_src_)),
			  groundtruth_src_(MakePlan(e.groundtruth_src_)),
			  grad_(MakePlan(e.grad_)),
			  num_keypoints_(e.num_keypoints_),
			  num_groundtruth_(e.num_groundtruth_),
			  ksize_x_(e.ksize_x_),
			  ksize_y_(e.ksize_y_),
			  channel_size_(e.channel_size_),
			  spatial_scale_(e.spatial_scale_),
			  src_height_(e.src_height_),
			  src_width_(e.src_width_) {}

		MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
			using namespace std;
			const index_t h = i % src_height_;
			const index_t w = j;
			const index_t c = (i / src_height_) % channel_size_;
			DType res = static_cast<DType>(0);
			for(index_t gt = 0; gt < num_groundtruth_; gt++)
			{
				index_t px = round(groundtruth_src_.Eval(gt, 0) * spatial_scale_);
				index_t py = round(groundtruth_src_.Eval(gt, 1) * spatial_scale_);
				for(index_t kp = 0; kp < num_keypoints_; kp++)
					for(index_t ox = 0; ox < ksize_x_; ox++)
						for(index_t oy = 0; oy < ksize_y_; oy++)
						{
							index_t x = round(keypoints_src_.Eval(px * src_width_ + py, kp * 2) * spatial_scale_) - ksize_x_ / 2 + ox;
							index_t y = round(keypoints_src_.Eval(px * src_width_ + py, kp * 2 + 1) * spatial_scale_) - ksize_y_ / 2 + oy;
							
							if(x == h && y == w)
								res += grad_.Eval(gt, kp * ksize_x_ * ksize_y_ * channel_size_ + 
										ox * ksize_y_ * channel_size_ + 
										oy * channel_size_ + c);
						}
			}
			return res;
		}

	private:
		Plan<DataExp, DType> keypoints_src_;
		Plan<GTExp, DType> groundtruth_src_, grad_;
		const index_t num_keypoints_, num_groundtruth_;
		const index_t ksize_x_, ksize_y_, channel_size_;
		const float spatial_scale_;
		const index_t src_height_, src_width_;
};
}
}
#endif
