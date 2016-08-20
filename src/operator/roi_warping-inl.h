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

namespace roiwarp_enum {
enum ROIWarpingOpInputs {kData, kROI, kGroundTruth};
enum ROIWarpingOpOutputs {kOut};
}

struct ROIWarpingParam : public dmlc::Parameter<ROIWarpingParam> {
	TShape warped_shape;
	float spatial_scale;
	DMLC_DECLARE_PARAMETER(ROIWarpingParam) {
		int shape[] = {1, 1};
		DMLC_DECLARE_FIELD(warped_shape).set_default(TShape(shape, shape + 2))
		.set_expect_ndim(2)
		.describe("warped_shape: for warping (y, x)");
        DMLC_DECLARE_FIELD(spatial_scale).set_default(1.0)
        .describe("spatial scale");
    }
};

template<typename xpu>
class ROIWarpingOp : public Operator {
    public:
        explicit ROIWarpingOp(ROIWarpingParam p) {
            this->param_ = p;
        }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(in_data.size(), 3) << "ROIWarpingOp Input: [data, roi, ground_truth]";
		CHECK_EQ(out_data.size(), 1) << "ROIWarpingOp Output: [output]";
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 4> data = in_data[roiwarp_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> out = out_data[roiwarp_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> rois = in_data[roiwarp_enum::kROI].get<xpu, 4, real_t>(s);
		Tensor<xpu, 2> gt = in_data[roiwarp_enum::kGroundTruth].get<xpu, 2, real_t>(s);
		
		mshadow::Shape<2> warped_shape = Shape2(param_.warped_shape[0],
												param_.warped_shape[1]);

		Assign(out, req[roiwarp_enum::kOut],
			   roi_warp(data, rois, gt, warped_shape, param_.spatial_scale));
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
		CHECK_EQ(in_data.size(), 2);
		CHECK_EQ(out_data.size(), 1);
		CHECK_EQ(req.size(), 2);
		CHECK_EQ(in_grad.size(), 2);
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 4> grad = out_grad[roiwarp_enum::kOut].get<xpu, 4, real_t>(s);
		Tensor<xpu, 4> data = in_data[roiwarp_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 2> rois = in_data[roiwarp_enum::kROI].get<xpu, 2, real_t>(s);
		Tensor<xpu, 4> data_grad = in_grad[roiwarp_enum::kData].get<xpu, 4, real_t>(s);
		Tensor<xpu, 2> rois_grad = in_grad[roiwarp_enum::kROI].get<xpu, 2, real_t>(s);
		
		mshadow::Shape<2> warped_shape = Shape2(param_.warped_shape[0],
												param_.warped_shape[1]);

		Assign(data_grad, req[roiwarp_enum::kData],
			   roi_unwarp_data(data, rois, grad, warped_shape, param_.spatial_scale));
		Assign(rois_grad, req[roiwarp_enum::kROI],
			   roi_unwarp_rois(data, rois, grad, warped_shape, param_.spatial_scale));
		*/
	}

	private:
		ROIWarpingParam param_;
};

template<typename xpu>
Operator *CreateOp(ROIWarpingParam param);

#if DMLC_USE_CXX11
class ROIWarpingProp : public OperatorProperty {
	public:
		std::vector<std::string> ListArguments() const override {
			return {"data", "rois", "ground truth"};
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
			CHECK_EQ(in_shape->size(), 3) << "Input: [data, rois, ground truth]";
			const TShape &dshape = in_shape->at(0);
			const TShape &rshape = in_shape->at(1);
			const TShape &gtshape = in_shape->at(2);
			CHECK_EQ(dshape.ndim(), 4) << \
				"ROIWarping: Input data should be 4D in (batch, channel, y, x)";
			CHECK_EQ(rshape.ndim(), 4) << \
				"ROIWarping: Input roi should be 4D in (batch, h, w, 4)";
			CHECK_EQ(gtshape.ndim(), 2) << \
				"ROIWarping: Input ground truth should be 2D in (num_gt, 2)";
			TShape oshape = dshape;
			if(dshape.ndim() == 0) return false;
			oshape[0] = gtshape[0];
			oshape[2] = param_.warped_shape[0];
			oshape[3] = param_.warped_shape[1];
			out_shape->clear();
			out_shape->push_back(oshape);
			return true;
		}

		OperatorProperty* Copy() const override {
			ROIWarpingProp *prop_sym = new ROIWarpingProp();
			prop_sym->param_ = this->param_;
			return prop_sym;
		}

		std::string TypeString() const override {
			return "ROIWarping";
		}

		std::vector<int> DeclareBackwardDependency(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data) const override {
			return {out_grad[roiwarp_enum::kOut], 
					in_data[roiwarp_enum::kData], 
					in_data[roiwarp_enum::kROI]};
		}

		std::vector<std::pair<int, void*> > BackwardInplaceOption(
				const std::vector<int> &out_grad,
				const std::vector<int> &in_data,
				const std::vector<int> &out_data,
				const std::vector<void*> &in_grad) const override {
			return {{in_data[roiwarp_enum::kData], in_grad[roiwarp_enum::kData]}};
		}

		Operator *CreateOperator(Context ctx) const override;

	private:
		ROIWarpingParam param_;
};

#endif
}
}

#endif
