#include "./face_element_sum-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FaceElementSumParam param) {
	return new FaceElementSumOp<cpu>(param);
}

Operator* FaceElementSumProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(FaceElementSumParam);

MXNET_REGISTER_OP_PROPERTY(FaceElementSum, FaceElementSumProp)
.describe("Face keypoints base + offset")
.add_argument("data", "Symbol", "offset")
.add_argument("keypoints", "Symbol", "keypoints")
.add_argument("groundtruth", "Symbol", "groundtruth")
.add_arguments(FaceElementSumParam::__FIELDS__());
}
}
