#include "./roi_pooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ROIPoolingParam param) {
	return new ROIPoolingOp<cpu>(param);
}

Operator* ROIPoolingProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(ROIPooling, ROIPoolingProp)
.describe("Region of interest pooling.")
.add_argument("data", "Symbol", "Input data to ROIPooling.")
.add_argument("rois", "Symbol", "Input label to ROIPooling.")
.add_arguments(ROIPoolingParam::__FIELDS__());
}
}
