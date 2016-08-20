#include "./rec2ellipse-inl.h"

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<gpu>(Rec2EllipseParam param) {
	return new Rec2EllipseOp<gpu>(param);
}

}
}
