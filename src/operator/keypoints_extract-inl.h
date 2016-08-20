#ifndef MXNET_OPERATOR_KEYPOINTS_EXTRACT_INL_H_
#define MXNET_OPERATOR_KEYPOINTS_EXTRACT_INL_H_

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

namespace keypointsextract_enum {
enum KeypointsExtractOpInputs {kData, kGroundTruth};
enum KeypointsExtractOpOutputs {kOut};
}

struct KeypointsExtractParam : public dmlc::Parameter<KeypointsExtractParam> {
	uint32_t anchor, num_keypoints;
	float spatial_scale;

	DMLC_DECLARE_PARAMETER(KeypointsExtractParam) {
		DMLC_DECLARE_FIELD(anchor).set_default(9)
		.describe("the index of the anchor");
		DMLC_DECLARE_FIELD(num_keypoints).set_default(10)
		.describe("the number of keypoints");
		DMLC_DECLARE_FIELD(spatial_scale).set_default(1.0)
		.describe("spatial scale");
	}
};

template<typename xpu>
class KeypointsExtractOp : public Operator {
	public:
		explicit KeypointsExtractOp(KeypointsExtractParam p) {
			this->param_ = p;
		}

		virtual void Forward(const OpContext &ctx,
							 const std::vector<TBlob> &in_data,
							 const std::vector<OpReqType> &req,
							 const std::vector<TBlob> &out_data,
							 const std::vector<TBlob> &aux_args) {
			using namespace mshadow;
			using namespace mshadow::expr;
			CHECK_EQ(in_data.size(), 2) << "KeypointsExtractOp Input: [Data GroundTruth]";
			CHECK_EQ(out_data.size(), 1) << "KeypointsExtractOp Output: [Output]";
			Stream<xpu> *s = ctx.get_stream<xpu>();
			Tensor<xpu, 4> data = in_data[keypointsextract_enum::kData].get<xpu, 4, real_t>(s);
			Tensor<xpu, 2> groundtruth = in_data[keypointsextract_enum::kGroundTruth].get<xpu, 2, real_t>(s);
			Tensor<xpu, 2> out = out_data[keypointsextract_enum::kOut].get<xpu, 2, real_t>(s);

			Assign(out, req[keypointsextract_enum::kOut],
				   keypoints_extract_forward(groundtruth, data,
											 param_.anchor, param_.num_keypoints,
											 param_.spatial_scale));
		}

		virtual void Backward(const OpContext &ctx,
							  const std::vector<TBlob> &out_grad,
							  const std::vector<TBlob> &in_data,
							  const std::vector<TBlob> &out_data,
							  const std::vector<OpReqType> &req,
							  const std::vector<TBlob> &in_grad,
							  const std::vector<TBlob> &aux_args) {
			/* This operator will back propagate nothing */
		}

	private:
		KeypointsExtractParam param_;
};

template<typename xpu>
Operator *CreateOp(KeypointsExtractParam param);

#if DMLC_USE_CXX11
class KeypointsExtractProp : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"data", "groundtruth"};
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
			CHECK_EQ(in_shape->size(), 2) << "Input: [data, groundtruth]";
			const TShape &dshape = in_shape->at(0);
			const TShape &gtshape = in_shape->at(1);
			CHECK_EQ(dshape.ndim(), 4) << \
				"ConfigPooling: Input data should be 4D in (batch, H, W, vector)";
			CHECK_EQ(gtshape.ndim(), 2) << \
				"ConfigPooling: Input ground truth should be 2D in (n, 2)";

			TShape oshape = gtshape;
			oshape[1] = param_.num_keypoints * 2;
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			KeypointsExtractProp *prop_sym = new KeypointsExtractProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "KeypointsExtract";
		}

		Operator *CreateOperator(Context ctx) const override;

	private:
		KeypointsExtractParam param_;
};

#endif
}
}

#endif
	

			

	




