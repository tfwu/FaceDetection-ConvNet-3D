#include "./face_element_sum-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(FaceElementSumParam param) {
	return new FaceElementSumOp<gpu>(param);
}

}
}
