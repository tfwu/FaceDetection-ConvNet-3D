#include "./box_predict-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(BoxPredictParam param) {
	return new BoxPredictOp<gpu>(param);
}

}
}
