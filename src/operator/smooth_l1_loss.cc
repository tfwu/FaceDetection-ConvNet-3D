#include "./smooth_l1_loss-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateSmoothL1LossOp<cpu>(SmoothL1LossParam param) {
    return new SmoothL1LossOp<cpu, mshadow_op::identity, mshadow_op::smooth_l1_loss_grad>(param);
}

Operator *SmoothL1LossProp::CreateOperator(Context ctx) const {
    DO_BIND_DISPATCH(CreateSmoothL1LossOp, param_);
}

DMLC_REGISTER_PARAMETER(SmoothL1LossParam);

MXNET_REGISTER_OP_PROPERTY(SmoothL1Loss, SmoothL1LossProp)
.describe("Use smooth L1 loss for loss calculation, this is usually used on final output of a net. Referred from fast RCNN.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("weight", "Symbol", "Input weight of each grid.")
.add_argument("label", "Symbol", "Input label to function.")
.add_arguments(SmoothL1LossParam::__FIELDS__());
}
}
