#ifndef MXNET_OPERATOR_GEN_ELL_LABEL_INL_H_
#define MXNET_OPERATOR_GEN_ELL_LABEL_INL_H_

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

namespace genelllabel_enum {
enum GenEllLabelOpInputs {kRoi, kLabel, kGroundTruth};
enum GenEllLabelOpOutputs {kOut};
}

struct GenEllLabelParam : public dmlc::Parameter<GenEllLabelParam> {
	float spatial_scale;
	DMLC_DECLARE_PARAMETER(GenEllLabelParam) {
		DMLC_DECLARE_FIELD(spatial_scale).set_default(1.0)
		.describe("spatial scale");
	}
};

template<typename xpu>
class GenEllLabelOp : public Operator {
    public:
        explicit GenEllLabelOp(GenEllLabelParam p) {
            this->param_ = p;
        }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(in_data.size(), 3) << "GenEllLabelOp Input: [rois, label, gt]";
		CHECK_EQ(out_data.size(), 1) << "GenEllLabelOp Output: [output]";
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 4> roi = in_data[genelllabel_enum::kRoi].get<xpu, 4, real_t>(s);
		Tensor<xpu, 2> gt = in_data[genelllabel_enum::kGroundTruth].get<xpu, 2, real_t>(s);
		Tensor<xpu, 2> label = in_data[genelllabel_enum::kLabel].get<xpu, 2, real_t>(s);
		Tensor<xpu, 2> out = out_data[genelllabel_enum::kOut].get<xpu, 2, real_t>(s);
		
		Assign(out, req[genelllabel_enum::kOut], gen_ell_label_forward(roi, label, gt, param_.spatial_scale));
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
	}

	private:
		GenEllLabelParam param_;
};

template<typename xpu>
Operator *CreateOp(GenEllLabelParam param);

#if DMLC_USE_CXX11
class GenEllLabelProp : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"rois", "label", "ground truth"};
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
			CHECK_EQ(in_shape->size(), 3) << "Input: [rois, label, gt]";
			const TShape &rshape = in_shape->at(0);
			const TShape &lshape = in_shape->at(1);
			const TShape &gtshape = in_shape->at(2);
			CHECK_EQ(rshape.ndim(), 4) << \
				"GenEllLabel: Input rois should be 4D in (batch, h, w, 4)";
			CHECK_EQ(lshape.ndim(), 2) << \
				"GenEllLabel: Input label should be 2D in (num_roi, 5)";
			CHECK_EQ(gtshape.ndim(), 2) << \
				"GenEllLabel: Input gt should be 2D in (num_gt, 2)";
			TShape oshape = lshape;
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			GenEllLabelProp *prop_sym = new GenEllLabelProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "GenEllLabel";
		}

		std::vector<int> DeclareBackwardDependency(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data) const override {
			return {};
		}

		std::vector<std::pair<int, void*> > BackwardInplaceOption(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data,
				const std::vector<void*> &in_grad) const override {
			return {{in_data[genelllabel_enum::kRoi], in_grad[genelllabel_enum::kRoi]}};
		}

		Operator *CreateOperator(Context ctx) const override;

	private:
		GenEllLabelParam param_;
};

#endif
}
}

#endif
