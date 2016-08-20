#include "./keypoints_extract-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(KeypointsExtractParam param) {
	return new KeypointsExtractOp<cpu>(param);
}

Operator* KeypointsExtractProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(KeypointsExtractParam);

MXNET_REGISTER_OP_PROPERTY(KeypointsExtract, KeypointsExtractProp)
.describe("keypoints extract")
.add_argument("data", "Symbol", "Projected keypoints")
.add_argument("ground_truth", "Symbol", "ground_truth")
.add_arguments(KeypointsExtractParam::__FIELDS__());
}
}


