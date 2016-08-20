#ifndef MXNET_OPERATOR_CONFIG_POOLING_INL_H_
#define MXNET_OPERATOR_CONFIG_POOLING_INL_H_

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

namespace configpooling_enum {
enum ConfigPoolingOpInputs {kData, kKeyPoints, kGroundTruth};
enum ConfigPoolingOpOutputs {kOut};
}

struct ConfigPoolingParam : public dmlc::Parameter<ConfigPoolingParam> {
	uint32_t num_keypoints;
	TShape kernel;
	float spatial_scale;

	DMLC_DECLARE_PARAMETER(ConfigPoolingParam) {
		int shape[] = {1, 1};
		DMLC_DECLARE_FIELD(num_keypoints).set_default(10)
		.describe("the number of facial keypoints");
		DMLC_DECLARE_FIELD(kernel).set_default(TShape(shape, shape + 2))
		.describe("the shape of the kernel");
		DMLC_DECLARE_FIELD(spatial_scale).set_default(1.0)
		.describe("spatial scale");
	}
};

template<typename xpu>
class ConfigPoolingOp: public Operator {
	public:
		explicit ConfigPoolingOp(ConfigPoolingParam p) {
			this->param_ = p;
		}
	
		virtual void Forward(const OpContext &ctx,
							 const std::vector<TBlob> &in_data,
							 const std::vector<OpReqType> &req,
							 const std::vector<TBlob> &out_data,
							 const std::vector<TBlob> &aux_args) {
			using namespace mshadow;
			using namespace mshadow::expr;
			CHECK_EQ(in_data.size(), 3) << "ConfigPoolingOp Input: [Data KeyPoints GroundTruth]";
			CHECK_EQ(out_data.size(), 1) << "ConfigPoolingOp Output: [Output]";
			Stream<xpu> *s = ctx.get_stream<xpu>();
			Tensor<xpu, 4> data = in_data[configpooling_enum::kData].get<xpu, 4, real_t>(s);
			Tensor<xpu, 4> keypoints = in_data[configpooling_enum::kKeyPoints].get<xpu, 4, real_t>(s);
			Tensor<xpu, 2> groundtruth = in_data[configpooling_enum::kGroundTruth].get<xpu, 2, real_t>(s);
			Tensor<xpu, 2> out = out_data[configpooling_enum::kOut].get<xpu, 2, real_t>(s);

			Assign(out, req[configpooling_enum::kOut],
				   configpool_forward(data, keypoints, groundtruth,
									  param_.num_keypoints,
									  param_.kernel[0],
									  param_.kernel[1],
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
			CHECK_EQ(in_data.size(), 3);
			CHECK_EQ(out_data.size(), 1);
			CHECK_EQ(req.size(), 3);
			CHECK_EQ(in_grad.size(), 3);
			Stream<xpu> *s = ctx.get_stream<xpu>();
			Tensor<xpu, 2> grad = out_grad[configpooling_enum::kOut].get<xpu, 2, real_t>(s);
			Tensor<xpu, 4> data = in_data[configpooling_enum::kData].get<xpu, 4, real_t>(s);
			Tensor<xpu, 4> keypoints = in_data[configpooling_enum::kKeyPoints].get<xpu, 4, real_t>(s);
			Tensor<xpu, 2> groundtruth = in_data[configpooling_enum::kGroundTruth].get<xpu, 2, real_t>(s);
			Tensor<xpu, 4> input_grad = in_grad[configpooling_enum::kData].get<xpu, 4, real_t>(s);
			
			Assign(input_grad, req[configpooling_enum::kData],
				   configpool_backward(data, keypoints, groundtruth, grad,
									   param_.num_keypoints,
									   param_.kernel[0],
									   param_.kernel[1],
									   param_.spatial_scale));
		}

	private:
		ConfigPoolingParam param_;
};

template<typename xpu>
Operator *CreateOp(ConfigPoolingParam param);

#if DMLC_USE_CXX11
class ConfigPoolingProp : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"data", "keypoints", "groundtruth"};
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
			CHECK_EQ(in_shape->size(), 3) << "Input: [data, keypoints, groundtruth]";
			const TShape &dshape = in_shape->at(0);
			const TShape &kshape = in_shape->at(1);
			const TShape &gtshape = in_shape->at(2);
			CHECK_EQ(dshape.ndim(), 4) << \
				"ConfigPooling: Input data should be 4D in (batch, channel, y, x)";
			CHECK_EQ(kshape.ndim(), 4) << \
				"ConfigPooling: Input keypoints should be 4D in (batch, H, W, num_keypoints * 2)";
			CHECK_EQ(gtshape.ndim(), 2) << \
				"ConfigPooling: Input ground truth should be 2D in (n, 2)";

			TShape oshape = gtshape;
			oshape[1] = param_.num_keypoints * param_.kernel[0] * param_.kernel[1] * dshape[1];
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			ConfigPoolingProp *prop_sym = new ConfigPoolingProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "ConfigPooling";
		}

		std::vector<int> DeclareBackwardDependency(
			const std::vector<int> &out_grad,
			const std::vector<int> &in_data,
			const std::vector<int> &out_data) const override {
			return {out_grad[configpooling_enum::kOut],
					in_data[configpooling_enum::kData],
					in_data[configpooling_enum::kKeyPoints],
					in_data[configpooling_enum::kGroundTruth]};
		}

		std::vector<std::pair<int, void*> > BackwardInplaceOption(
			const std::vector<int> &out_grad,
			const std::vector<int> &in_data,
			const std::vector<int> &out_data,
			const std::vector<void*> &in_grad) const override {
			return {{in_data[configpooling_enum::kData], in_grad[configpooling_enum::kData]}};
		}

		Operator *CreateOperator(Context ctx) const override;

	private:
		ConfigPoolingParam param_;
};

#endif
}
}

#endif
