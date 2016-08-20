#ifndef MXNET_OPERATOR_ROI_POOLING_INL_H_
#define MXNET_OPERATOR_ROI_POOLING_INL_H_

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

namespace roipool_enum {
enum ROIPoolingOpInputs {kData, kROI};
enum ROIPoolingOpOutputs {kOut};
}

struct ROIPoolingParam : public dmlc::Parameter<ROIPoolingParam> {
	uint32_t num_rois;
	TShape pooled_shape;
	float spatial_scale;

	DMLC_DECLARE_PARAMETER(ROIPoolingParam) {
		DMLC_DECLARE_FIELD(num_rois).set_default(1)
		.describe("the number of rois per image");

		int shape[] = {1, 1};
		DMLC_DECLARE_FIELD(pooled_shape).set_default(TShape(shape, shape + 2))
		.set_expect_ndim(2)
		.describe("pooled_shape: for pooling (y, x)");

        DMLC_DECLARE_FIELD(spatial_scale).set_default(1)
        .describe("spatial scale");
    }
};

template<typename xpu>
class ROIPoolingOp : public Operator {
    public:
        explicit ROIPoolingOp(ROIPoolingParam p) {
            this->param_ = p;
        }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(in_data.size(), 2) << "ROIPoolingOp Input: [data, roi]";
		CHECK_EQ(out_data.size(), 1) << "ROIPoolingOp Output: [output]";
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 4> data = in_data[roipool_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> out = out_data[roipool_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 2> rois = in_data[roipool_enum::kROI].get<xpu, 2, real_t>(s);
		
		mshadow::Shape<2> pooled_shape = Shape2(param_.pooled_shape[0],
												param_.pooled_shape[1]);

		// Shape Check:
		// printf("%u %u %u %u\n", data.shape_[0], data.shape_[1], data.shape_[2], data.shape_[3]);
		// printf("%u %u\n", rois.shape_[0], rois.shape_[1]);
	
		Assign(out, req[roipool_enum::kOut],
			   roipool(data, rois,
					   param_.num_rois,
					   pooled_shape,
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
		Tensor<xpu, 4> grad = out_grad[roipool_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> data = in_data[roipool_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 2> rois = in_data[roipool_enum::kROI].get<xpu, 2, real_t>(s);
		Tensor<xpu, 4> output_data = out_data[roipool_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> input_grad = in_grad[roipool_enum::kData].get<xpu, 4, real_t>(s);
		
		mshadow::Shape<2> pooled_shape = Shape2(param_.pooled_shape[0],
												param_.pooled_shape[1]);

		// Shape Check:
		// printf("%u %u %u %u\n", grad.shape_[0], grad.shape_[1], grad.shape_[2], grad.shape_[3]);
		// printf("%u %u %u %u\n", input_grad.shape_[0], input_grad.shape_[1], input_grad.shape_[2], input_grad.shape_[3]);

		Assign(input_grad, req[roipool_enum::kData],
			   roiunpool(data, rois,
						 output_data, grad,
						 param_.num_rois,
						 pooled_shape,
						 param_.spatial_scale));
	}

	private:
		ROIPoolingParam param_;
};

template<typename xpu>
Operator *CreateOp(ROIPoolingParam param);

#if DMLC_USE_CXX11
class ROIPoolingProp : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"data", "rois"};
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
			CHECK_EQ(in_shape->size(), 2) << "Input: [data, rois]";
			const TShape &dshape = in_shape->at(0);
			CHECK_EQ(dshape.ndim(), 4) << \
				"ROIPooling: Input data should be 4D in (batch, channel, y, x)";
			TShape oshape = dshape;
			if(dshape.ndim() == 0) return false;

			oshape[0] = param_.num_rois * dshape[0];
			oshape[2] = param_.pooled_shape[0];
			oshape[3] = param_.pooled_shape[1];
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			ROIPoolingProp *prop_sym = new ROIPoolingProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "ROIPooling";
		}

		std::vector<int> DeclareBackwardDependency(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data) const override {
			return {out_grad[roipool_enum::kOut], in_data[roipool_enum::kData], in_data[roipool_enum::kROI], out_data[roipool_enum::kOut]};
		}

		std::vector<std::pair<int, void*> > BackwardInplaceOption(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data,
				const std::vector<void*> &in_grad) const override {
			return {{in_data[roipool_enum::kData], in_grad[roipool_enum::kData]}};
		}

		Operator *CreateOperator(Context ctx) const override;

	private:
		ROIPoolingParam param_;
};

#endif
}
}

#endif
