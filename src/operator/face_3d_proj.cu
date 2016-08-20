#include "./face_3d_proj-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(Face3DProjParam param) {
	return new Face3DProjOp<gpu>(param);
}

}
}
