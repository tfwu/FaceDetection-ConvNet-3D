#include "./config_pooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(ConfigPoolingParam param) {
	return new ConfigPoolingOp<gpu>(param);
}

}
}
