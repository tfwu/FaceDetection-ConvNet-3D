#include "./smooth_l1_loss-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateSmoothL1LossOp<gpu>(SmoothL1LossParam param) {
    return new SmoothL1LossOp<gpu, mshadow_op::identity, mshadow_op::smooth_l1_loss_grad>(param);
}

}
}
