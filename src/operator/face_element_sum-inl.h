#ifndef MXNET_OPERATOR_FACE_ELEMENT_SUM_INL_H_
#define MXNET_OPERATOR_FACE_ELEMENT_SUM_INL_H_

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

namespace faceelementsum_enum {
enum FaceElementSumOpInputs {kData, kKeyPoints, kGroundTruth};
enum FaceElementSumOpOutputs {kOut};
}

struct FaceElementSumParam : public dmlc::Parameter<FaceElementSumParam> {
	uint32_t num_keypoints;
	float spatial_scale;

	DMLC_DECLARE_PARAMETER(FaceElementSumParam) {
		DMLC_DECLARE_FIELD(num_keypoints).set_default(10)
		.describe("the number of facial keypoints");
		DMLC_DECLARE_FIELD(spatial_scale).set_default(1.0)
		.describe("spatial scale");
	}
};

template<typename xpu>
class FaceElementSumOp : public Operator {
	public:
		explicit FaceElementSumOp(FaceElementSumParam p) {
			this->param_ = p;
		}

		virtual void Forward(const OpContext &ctx,
							 const std::vector<TBlob> &in_data,
							 const std::vector<OpReqType> &req,
							 const std::vector<TBlob> &out_data,
							 const std::vector<TBlob> &aux_args) {
			using namespace mshadow;
			using namespace mshadow::expr;
			CHECK_EQ(in_data.size(), 3) << "FaceElementSumOp Input: [Data KeyPoints GroundTruth]";
			CHECK_EQ(out_data.size(), 1) << "FaceElementSumOp Output: [Output]";
			Stream<xpu> *s = ctx.get_stream<xpu>();
			Tensor<xpu, 2> data = in_data[faceelementsum_enum::kData].get<xpu, 2, real_t>(s);
			Tensor<xpu, 4> keypoints = in_data[faceelementsum_enum::kKeyPoints].get<xpu, 4, real_t>(s);
			Tensor<xpu, 2> groundtruth = in_data[faceelementsum_enum::kGroundTruth].get<xpu, 2, real_t>(s);
			Tensor<xpu, 2> out = out_data[faceelementsum_enum::kOut].get<xpu, 2, real_t>(s);

			Assign(out, req[faceelementsum_enum::kOut],
				   face_element_sum_forward(data, keypoints, groundtruth,
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
			CHECK_EQ(in_data.size(), 3);
			CHECK_EQ(out_data.size(), 1);
			CHECK_EQ(req.size(), 3);
			CHECK_EQ(in_grad.size(), 3);
			Stream<xpu> *s = ctx.get_stream<xpu>();
			Tensor<xpu, 2> grad = out_grad[faceelementsum_enum::kOut].FlatTo2D<xpu, real_t>(s);
			Tensor<xpu, 2> input_grad = in_grad[faceelementsum_enum::kData].FlatTo2D<xpu, real_t>(s);
			
			Assign(input_grad, req[faceelementsum_enum::kData], F<mshadow_op::identity>(grad));
		}

	private:
		FaceElementSumParam param_;
};

template<typename xpu>
Operator *CreateOp(FaceElementSumParam param);

#if DMLC_USE_CXX11
class FaceElementSumProp : public OperatorProperty {
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
			CHECK_EQ(dshape.ndim(), 2) << \
				"FaceElementSum: Input data should be 2D in (batch, vector)";
			CHECK_EQ(kshape.ndim(), 4) << \
				"FaceElementSum: Input keypoints should be 4D in (batch, H, W, num_keypoints * 2)";
			CHECK_EQ(gtshape.ndim(), 2) << \
				"FaceElementSum: Input ground truth should be 2D in (n, 2)";

			TShape oshape = dshape;
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			FaceElementSumProp *prop_sym = new FaceElementSumProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "FaceElementSum";
		}

		std::vector<int> DeclareBackwardDependency(
			const std::vector<int> &out_grad,
			const std::vector<int> &in_data,
			const std::vector<int> &out_data) const override {
			return out_grad;
		}

		std::vector<std::pair<int, void*> > BackwardInplaceOption(
			const std::vector<int> &out_grad,
			const std::vector<int> &in_data,
			const std::vector<int> &out_data,
			const std::vector<void*> &in_grad) const override {
			return {{out_grad[0], in_grad[0]}};
		}

		std::vector<std::pair<int, void*> > ForwardInplaceOption(
			const std::vector<int> &in_data,
			const std::vector<void*> &out_data) const override {
			return {{in_data[0], out_data[0]}};
		}
		
		Operator *CreateOperator(Context ctx) const override;

	private:
		FaceElementSumParam param_;
};

#endif
}
}

#endif
	

			

		



	

