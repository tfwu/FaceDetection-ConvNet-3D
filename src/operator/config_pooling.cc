#include "./config_pooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ConfigPoolingParam param) {
	return new ConfigPoolingOp<cpu>(param);
}

Operator* ConfigPoolingProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ConfigPoolingParam);

MXNET_REGISTER_OP_PROPERTY(ConfigPooling, ConfigPoolingProp)
.describe("Configuration pooling layer.")
.add_argument("data", "Symbol", "Feature map")
.add_argument("kpoints", "Symbol", "Projected keypoint")
.add_argument("groundtruth", "Symbole", "Ground truth")
.add_arguments(ConfigPoolingParam::__FIELDS__());
}
}

