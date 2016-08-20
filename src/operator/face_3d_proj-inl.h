#ifndef MXNET_OPERATOR_FACE_3D_PROJ_INL_H_
#define MXNET_OPERATOR_FACE_3D_PROJ_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace face3dproj_enum {
enum Face3DProjOpInputs {kData, kMFace};
enum Face3DProjOpOutputs {kOut};
}

struct Face3DProjParam : public dmlc::Parameter<Face3DProjParam> {
	uint32_t num_keypoints;
	float spatial_scale;

	DMLC_DECLARE_PARAMETER(Face3DProjParam) {
		DMLC_DECLARE_FIELD(num_keypoints).set_default(10)
		.describe("the number of facial keypoints");
		DMLC_DECLARE_FIELD(spatial_scale).set_default(1.0)
		.describe("spatial scale");
	}
};

template<typename xpu>
class Face3DProjOp: public Operator {
	public:
		explicit Face3DProjOp(Face3DProjParam p) {
			this->param_ = p;
		}
	
	virtual void Forward(const OpContext &ctx,
						 const std::vector<TBlob> &in_data,
						 const std::vector<OpReqType> &req,
						 const std::vector<TBlob> &out_data,
						 const std::vector<TBlob> &aux_args) {
		using namespace mshadow;
		using namespace mshadow::expr;
		CHECK_EQ(in_data.size(), 2) << "Face3DProjOp Input: [data mface]";
		CHECK_EQ(out_data.size(), 1) << "Face3DProjOp Output: [output]";
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 4> data = in_data[face3dproj_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> out = out_data[face3dproj_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 2> mface = in_data[face3dproj_enum::kMFace].get<xpu, 2, real_t>(s);

		Assign(out, req[face3dproj_enum::kOut],
			   face3dproj_forward(data, mface,
								  param_.num_keypoints,
								  param_.spatial_scale));
	}

	virtual void Backward(const OpContext &ctx,
						  const std::vector<TBlob> &out_grad,
						  const std::vector<TBlob> &in_data,
						  const std::vector<TBlob> &out_data,
						  const std::vector<OpReqType> &req,
						  const std::vector<TBlob> &in_grad,
						  const std::vector<TBlob> &aux_args) {
		using namespace mshadow;
		using namespace mshadow::expr;
		CHECK_EQ(out_grad.size(), 1);
		CHECK_EQ(in_data.size(), 2);
		CHECK_EQ(out_data.size(), 1);
		CHECK_EQ(req.size(), 2);
		CHECK_EQ(in_grad.size(), 2);
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 4> grad = out_grad[face3dproj_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> data = in_data[face3dproj_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 2> mface = in_data[face3dproj_enum::kMFace].get<xpu, 2, real_t>(s);
		Tensor<xpu, 4> input_grad = in_grad[face3dproj_enum::kData].get<xpu, 4, real_t>(s);

		Assign(input_grad, req[face3dproj_enum::kData],
			   face3dproj_backward(data, mface, grad,
								   param_.num_keypoints));
	}

	private:
		Face3DProjParam param_;
};

template<typename xpu>
Operator *CreateOp(Face3DProjParam param);

#if DMLC_USE_CXX11
class Face3DProjProp : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"data", "mface"};
		}

		void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
			param_.Init(kwargs);
		}

		std::map<std::string, std::string> GetParams() const override {
			return param_.__DICT__();
		}

		bool InferShape(std::vector<TShape> *in_shape,
						std::vector<TShape> *out_shape,
						std::vector<TShape> *aux_shape) const override {
			CHECK_EQ(in_shape->size(), 2) << "Input: [data, mface]";
			const TShape &dshape = in_shape->at(0);
			const TShape &fshape = in_shape->at(1);
			CHECK_EQ(dshape.ndim(), 4) << \
				"Face3DProj: Input data should be 4D in (batch, channel, y, x)";
			CHECK_EQ(fshape.ndim(), 2) << \
				"Face3DProj: Input mean face should be 2D in (num_keypoints, 3)";

			TShape oshape = dshape;
			oshape[1] = dshape[2];
			oshape[2] = dshape[3];
			oshape[3] = param_.num_keypoints * 2;
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			Face3DProjProp *prop_sym = new Face3DProjProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "Face3DProj";
		}

		std::vector<int> DeclareBackwardDependency(
			const std::vector<int> &out_grad,
			const std::vector<int> &in_data,
			const std::vector<int> &out_data) const override {
			return {out_grad[face3dproj_enum::kOut], 
					in_data[face3dproj_enum::kData],
					in_data[face3dproj_enum::kMFace]};
		}

		std::vector<std::pair<int, void*> > BackwardInplaceOption(
			const std::vector<int> &out_grad,
			const std::vector<int> &in_data,
			const std::vector<int> &out_data,
			const std::vector<void*> &in_grad) const override {
			return {{in_data[face3dproj_enum::kData], in_grad[face3dproj_enum::kData]}};
		}

		Operator *CreateOperator(Context ctx) const override;
	
	private:
		Face3DProjParam param_;
};

#endif
}
}

#endif
