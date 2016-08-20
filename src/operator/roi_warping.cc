#include "./roi_warping-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ROIWarpingParam param) {
	return new ROIWarpingOp<cpu>(param);
}

Operator* ROIWarpingProp::CreateOperator(Context ctx) const {
	DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ROIWarpingParam);

MXNET_REGISTER_OP_PROPERTY(ROIWarping, ROIWarpingProp)
.describe("Region of interest warping.")
.add_argument("data", "Symbol", "Input data to ROIWarping.")
.add_argument("rois", "Symbol", "Input label to ROIWarping.")
.add_argument("gt", "Symbol", "Input ground truth to ROIWarping.")
.add_arguments(ROIWarpingParam::__FIELDS__());
}
}
