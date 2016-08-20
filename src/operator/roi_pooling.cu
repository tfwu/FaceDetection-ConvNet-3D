#include "./roi_pooling-inl.h"

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<gpu>(ROIPoolingParam param) {
	return new ROIPoolingOp<gpu>(param);
}

}
}
