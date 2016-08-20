#ifndef MXNET_OPERATOR_BOX_PREDICT_INL_H_
#define MXNET_OPERATOR_BOX_PREDICT_INL_H_

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

namespace boxpredict_enum {
enum BoxPredictOpInputs {kData};
enum BoxPredictOpOutputs {kOut};
}

struct BoxPredictParam : public dmlc::Parameter<BoxPredictParam> {
	DMLC_DECLARE_PARAMETER(BoxPredictParam) {
	}
};

template<typename xpu>
class BoxPredictOp: public Operator {
	public:
		explicit BoxPredictOp(BoxPredictParam p) {
			this->param_ = p;
		}
	
	virtual void Forward(const OpContext &ctx,
						 const std::vector<TBlob> &in_data,
						 const std::vector<OpReqType> &req,
						 const std::vector<TBlob> &out_data,
						 const std::vector<TBlob> &aux_args) {
		using namespace mshadow;
		using namespace mshadow::expr;
		CHECK_EQ(in_data.size(), 1) << "BoxPredictOp Input: [data]";
		CHECK_EQ(out_data.size(), 1) << "BoxPredictOp Output: [output]";
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 4> data = in_data[boxpredict_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> out = out_data[boxpredict_enum::kOut].get<xpu, 4, real_t>(s);

		Assign(out, req[boxpredict_enum::kOut], box_predict_forward(data));
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
		/*
		CHECK_EQ(out_grad.size(), 1);
		CHECK_EQ(in_data.size(), 1);
		CHECK_EQ(out_data.size(), 1);
		CHECK_EQ(req.size(), 1);
		CHECK_EQ(in_grad.size(), 1);
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 4> grad = out_grad[boxpredict_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> data = in_data[boxpredict_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> out = out_data[boxpredict_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> input_grad = in_grad[boxpredict_enum::kData].get<xpu, 4, real_t>(s);

		Assign(input_grad, req[boxpredict_enum::kData], box_predict_backward(data, out, grad));
		*/
	}

	private:
		BoxPredictParam param_;
};

template<typename xpu>
Operator *CreateOp(BoxPredictParam param);

#if DMLC_USE_CXX11
class BoxPredictProp : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"data"};
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
			CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
			const TShape &dshape = in_shape->at(0);
			CHECK_EQ(dshape.ndim(), 4) << \
				"BoxPredict: Input data should be 4D in (batch, channel, y, x)";
			TShape oshape = dshape;
			oshape[3] = 4;
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			BoxPredictProp *prop_sym = new BoxPredictProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "BoxPredict";
		}

		std::vector<int> DeclareBackwardDependency(
			const std::vector<int> &out_grad,
			const std::vector<int> &in_data,
			const std::vector<int> &out_data) const override {
			return {out_grad[boxpredict_enum::kOut],
					out_data[boxpredict_enum::kOut],
					in_data[boxpredict_enum::kData]};
		}

		std::vector<std::pair<int, void*> > BackwardInplaceOption(
			const std::vector<int> &out_grad,
			const std::vector<int> &in_data,
			const std::vector<int> &out_data,
			const std::vector<void*> &in_grad) const override {
			return {{in_data[boxpredict_enum::kData], in_grad[boxpredict_enum::kData]}};
		}

		Operator *CreateOperator(Context ctx) const override;
	
	private:
		BoxPredictParam param_;
};

#endif
}
}

#endif



