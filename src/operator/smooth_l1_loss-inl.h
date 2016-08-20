#ifndef MXNET_OPERATOR_SMOOTH_L1_LOSS_INL_H_
#define MXNET_OPERATOR_SMOOTH_L1_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace mshadow_op {

/*! \brief smooth l1 loss operation */
struct smooth_l1_loss {
	template<typename DType>
	MSHADOW_XINLINE static DType Map(DType a, DType sigma) {
		// f(x) = 0.5 * (sigma * x)^2			if |x| < 1 / sigma / sigma
		//		= |x| - 0.5 / sigma / sigma		otherwise
		DType val = a;
		DType abs_val = fabs(a);
		DType sigma2 = sigma * sigma;
		if(abs_val < 1.0 / sigma2) {
			return DType(0.5 * val * val * sigma2);
		} else {
			return DType(abs_val - 0.5 / sigma2);
		}
	}
};

struct smooth_l1_loss_grad {
	template<typename DType>
	MSHADOW_XINLINE static DType Map(DType a, DType sigma) {
		// f'(x) = sigma * sigma * x	if |x| < 1 / sigma / sigma
		//		 = sign(x)				otherwise
		DType val = a;
		DType abs_val = std::fabs(a);
		DType sigma2 = sigma * sigma;
		if(abs_val < 1.0 / sigma2) {
			return DType(sigma2 * val);
		} else {
			return DType((DType(0) < val) - (val < DType(0)));
		}
	}
};

}	// namespace mshadow_op


namespace smoothl1_enum {
enum SmoothL1LossOpInputs {kData, kWeight, kLabel};
enum SmoothL1LossOutputs {kOut}; 
} // smooth_l1_enum

struct SmoothL1LossParam : public dmlc::Parameter<SmoothL1LossParam> {
    double sigma;
    DMLC_DECLARE_PARAMETER(SmoothL1LossParam) {
        DMLC_DECLARE_FIELD(sigma).set_default(1.0f)
        .describe("The parameter of smooth l1 loss.");
    }
};

/**
 * \brief This is the implementation of smooth l1 loss.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename ForwardOp, typename BackwardOp>
class SmoothL1LossOp : public Operator {
    public:
        explicit SmoothL1LossOp(SmoothL1LossParam p) {
            this->param_ = p;
        }

        virtual void Forward(const OpContext &ctx,
                             const std::vector<TBlob> &in_data,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &out_data,
                             const std::vector<TBlob> &aux_args) {
            using namespace mshadow;
            using namespace mshadow::expr;
            CHECK_EQ(in_data.size(), 3) << "SmoothL1LossOp Input: [data, weight, label]";
            CHECK_EQ(out_data.size(), 1) << "SmoothL1LossOp Output: [output]";
            Stream<xpu> *s = ctx.get_stream<xpu>();
            Tensor<xpu, 2> data = in_data[smoothl1_enum::kData].FlatTo2D<xpu, real_t>(s);
            Tensor<xpu, 2> out = out_data[smoothl1_enum::kOut].FlatTo2D<xpu, real_t>(s);
            Assign(out, req[smoothl1_enum::kOut], F<ForwardOp>(data));
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
            CHECK_EQ(in_data.size(), 3);
            CHECK_EQ(out_grad.size(), 1);
            CHECK_GE(in_grad.size(), 1);
            CHECK_GE(req.size(), 1);
            Stream<xpu> *s = ctx.get_stream<xpu>();
            real_t num_output = 
                in_data[smoothl1_enum::kLabel].Size() / in_data[smoothl1_enum::kLabel].shape_[0];
            Tensor<xpu, 2> out = out_data[smoothl1_enum::kOut].FlatTo2D<xpu, real_t>(s);
            Tensor<xpu, 2> grad = in_grad[smoothl1_enum::kData].FlatTo2D<xpu, real_t>(s);
            Tensor<xpu, 2> label = in_data[smoothl1_enum::kLabel]
                .get_with_shape<xpu, 2, real_t>(out.shape_, s);
			Tensor<xpu, 2> weight = in_data[smoothl1_enum::kWeight]
				.get_with_shape<xpu, 2, real_t>(out.shape_, s);

            Assign(grad, req[smoothl1_enum::kData], F<BackwardOp>(out - label, param_.sigma) * weight / num_output);
        }

    private:
        SmoothL1LossParam param_;
};

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateSmoothL1LossOp(SmoothL1LossParam param);

#if DMLC_USE_CXX11
class SmoothL1LossProp : public OperatorProperty {
    public:
        std::vector<std::string> ListArguments() const override {
            return {"data","weight", "label"};
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
            using namespace mshadow;
            CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, label]";
            const TShape &dshape = in_shape->at(0);
            if(dshape.ndim() == 0) return false;
            auto &lshape = (*in_shape)[1];
            if(lshape.ndim() == 0) {
                if (dshape.ndim() == 2 && dshape[1] == 1) {
                    lshape = Shape1(dshape[0]);
                } else {
                    lshape = dshape;
                }
            } else if (lshape[0] != dshape[0] || lshape.Size() != dshape.Size()) {
                std::ostringstream os;
                os << "Shape inconsistent, Provided " <<  '='<< lshape << ','
                    << " inferred shape=" << dshape;
                throw ::mxnet::op::InferShapeError(os.str(), 1);
            }
            out_shape->clear();
            out_shape->push_back(dshape);
            return true;
        }

        OperatorProperty* Copy() const override {
            auto ptr = new SmoothL1LossProp();
            ptr->param_ = param_;
            return ptr;
        }

        std::string TypeString() const override {
            return "SmoothL1Loss";
        }

        std::vector<int> DeclareBackwardDependency(
                const std::vector<int> &out_grad,
                const std::vector<int> &in_data,
                const std::vector<int> &out_data) const override {
            return {in_data[smoothl1_enum::kLabel], in_data[smoothl1_enum::kWeight],  out_data[smoothl1_enum::kOut]};
        }

        std::vector<std::pair<int, void*> > BackwardInplaceOption(
                const std::vector<int> &out_grad,
                const std::vector<int> &in_data,
                const std::vector<int> &out_data,
                const std::vector<void*> &in_grad) const override {
            return {{out_data[smoothl1_enum::kOut], in_grad[smoothl1_enum::kData]}};
        }

        std::vector<std::pair<int, void*> > ForwardInplaceOption(
                const std::vector<int> &in_data,
                const std::vector<void*> &out_data) const override {
            return {{in_data[smoothl1_enum::kData], out_data[smoothl1_enum::kOut]}};
        }

        Operator* CreateOperator(Context ctx) const override;

    private:
        SmoothL1LossParam param_;
};

#endif // DMLC_USE_CXX11
}   // namespace op
}   // namespace mxnet
#endif // MXNET_OPERATOR_SMOOTH_L1_LOSS_INL_H_



