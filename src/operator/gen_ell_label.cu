#include "./gen_ell_label-inl.h"

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<gpu>(GenEllLabelParam param) {
	return new GenEllLabelOp<gpu>(param);
}

}
}
