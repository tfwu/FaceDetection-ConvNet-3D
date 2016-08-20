#include "./rec2ellipse-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(Rec2EllipseParam param) {
	return new Rec2EllipseOp<cpu>(param);
}

Operator* Rec2EllipseProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(Rec2EllipseParam);

MXNET_REGISTER_OP_PROPERTY(Rec2Ellipse, Rec2EllipseProp)
.describe("Rec to ellipse.")
.add_argument("data", "Symbol", "Input data to Rec2Ellipse.")
.add_argument("offset", "Symbol", "Input label to Rec2Ellipse.")
.add_arguments(Rec2EllipseParam::__FIELDS__());
}
}
