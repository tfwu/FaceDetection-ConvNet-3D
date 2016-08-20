#include "./keypoints_extract-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(KeypointsExtractParam param) {
	return new KeypointsExtractOp<gpu>(param);
}

}
}
