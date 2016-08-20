import mxnet as mx


def get_vgg16_gen():

    relu_feature = mx.symbol.Variable(name="relu_feature")
    box_predict = mx.symbol.Variable(name="box_predict")
    ground_truth = mx.symbol.Variable(name="ground_truth")
    bbox_label = mx.symbol.Variable(name="bbox_label")

    ell_label = mx.symbol.GenEllLabel(*[box_predict, bbox_label, ground_truth], spatial_scale=0.5, name="ell_label")

    # roi warping
    roi_warping = mx.symbol.ROIWarping(*[relu_feature, box_predict, ground_truth], warped_shape=(28, 28),
                                       spatial_scale=0.5, name="roi_warping")
    roi_warping_pool = mx.symbol.Pooling(
            data=roi_warping, pool_type="max", kernel=(4, 4), stride=(4, 4), name="roi_warping_pool"
    )
    roi_warping_flatten = mx.symbol.Flatten(data=roi_warping_pool)

    loss_all = mx.symbol.Group([roi_warping_flatten, ell_label])

    return loss_all
