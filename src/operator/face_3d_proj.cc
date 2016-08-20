#include "./face_3d_proj-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(Face3DProjParam param) {
	return new Face3DProjOp<cpu>(param);
}

Operator* Face3DProjProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(Face3DProjParam);

MXNET_REGISTER_OP_PROPERTY(Face3DProj, Face3DProjProp)
.describe("Face 3d projection layer.")
.add_argument("data", "Symbol", "Input data to Face3DProj.")
.add_argument("mface", "Symbol", "Mean face.")
.add_arguments(Face3DProjParam::__FIELDS__());
}
}
