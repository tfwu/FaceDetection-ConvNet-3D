#ifndef MXNET_OPERATOR_REC2ELLIPSE_INL_H_
#define MXNET_OPERATOR_REC2ELLIPSE_INL_H_

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

namespace rec2ell_enum {
enum Rec2EllipseOpInputs {kOff, kRoi};
enum Rec2EllipseOpOutputs {kOut};
}

struct Rec2EllipseParam : public dmlc::Parameter<Rec2EllipseParam> {
	float spatial_scale;
	DMLC_DECLARE_PARAMETER(Rec2EllipseParam) {
        DMLC_DECLARE_FIELD(spatial_scale).set_default(1.0)
        .describe("spatial scale");
    }
};

template<typename xpu>
class Rec2EllipseOp : public Operator {
    public:
        explicit Rec2EllipseOp(Rec2EllipseParam p) {
            this->param_ = p;
        }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(in_data.size(), 2) << "Rec2EllipseOp Input: [offset, rois]";
		CHECK_EQ(out_data.size(), 1) << "Rec2EllipseOp Output: [output]";
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 2> off = in_data[rec2ell_enum::kOff].get<xpu, 2, real_t>(s);
		Tensor<xpu, 2> roi = in_data[rec2ell_enum::kRoi].get<xpu, 2, real_t>(s);
		Tensor<xpu, 2> out = out_data[rec2ell_enum::kOut].get<xpu, 2, real_t>(s);
		
		Assign(out, req[rec2ell_enum::kOut],
			   rec2ell_forward(off, roi, param_.spatial_scale));
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
		Tensor<xpu, 2> grad = out_grad[rec2ell_enum::kOut].get<xpu, 2, real_t>(s);
		Tensor<xpu, 2> off = in_data[rec2ell_enum::kOff].get<xpu, 2, real_t>(s);
		Tensor<xpu, 2> roi = in_data[rec2ell_enum::kRoi].get<xpu, 2, real_t>(s);
		Tensor<xpu, 2> off_grad = in_grad[rec2ell_enum::kOff].get<xpu, 2, real_t>(s);
		
		Assign(off_grad, req[rec2ell_enum::kOff],
			   rec2ell_backward(off, roi, grad, param_.spatial_scale));
	}

	private:
		Rec2EllipseParam param_;
};

template<typename xpu>
Operator *CreateOp(Rec2EllipseParam param);

#if DMLC_USE_CXX11
class Rec2EllipseProp : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"offset", "rois"};
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
			CHECK_EQ(in_shape->size(), 2) << "Input: [offset, rois]";
			const TShape &dshape = in_shape->at(0);
			const TShape &rshape = in_shape->at(1);
			CHECK_EQ(dshape.ndim(), 2) << \
				"Rec2Ellipse: Input offset should be 2D in (num_roi, 5)";
			CHECK_EQ(rshape.ndim(), 2) << \
				"Rec2Ellipse: Input offset should be 2D in (num_roi, 4)";
			TShape oshape = dshape;
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			Rec2EllipseProp *prop_sym = new Rec2EllipseProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "Rec2Ellipse";
		}

		std::vector<int> DeclareBackwardDependency(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data) const override {
			return {out_grad[rec2ell_enum::kOut], 
					in_data[rec2ell_enum::kOff], 
					in_data[rec2ell_enum::kRoi]};
		}

		std::vector<std::pair<int, void*> > BackwardInplaceOption(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data,
				const std::vector<void*> &in_grad) const override {
			return {{in_data[rec2ell_enum::kOff], in_grad[rec2ell_enum::kOff]}};
		}

		Operator *CreateOperator(Context ctx) const override;

	private:
		Rec2EllipseParam param_;
};

#endif
}
}

#endif
