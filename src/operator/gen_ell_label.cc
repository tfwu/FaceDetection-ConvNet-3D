#include "./gen_ell_label-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(GenEllLabelParam param) {
	return new GenEllLabelOp<cpu>(param);
}

Operator* GenEllLabelProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(GenEllLabelParam);

MXNET_REGISTER_OP_PROPERTY(GenEllLabel, GenEllLabelProp)
.describe("Generate ellipse label.")
.add_argument("roi", "Symbol", "Input roi.")
.add_argument("label", "Symbol", "Input label.")
.add_argument("ground truth", "Symbol", "Input ground truth")
.add_arguments(GenEllLabelParam::__FIELDS__());
}
}
