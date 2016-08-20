#include "./roi_warping-inl.h"

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<gpu>(ROIWarpingParam param) {
	return new ROIWarpingOp<gpu>(param);
}

}
}
