#include "./box_predict-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(BoxPredictParam param) {
	return new BoxPredictOp<cpu>(param);
}

Operator* BoxPredictProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(BoxPredictParam);

MXNET_REGISTER_OP_PROPERTY(BoxPredict, BoxPredictProp)
.describe("Convert keypoints into bounding box")
.add_argument("data", "Symbol", "position of predicted keypoints")
.add_arguments(BoxPredictParam::__FIELDS__());
}
}
