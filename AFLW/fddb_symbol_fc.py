import mxnet as mx


def get_vgg16_fc():

    relu_feature = mx.symbol.Variable(name="relu_feature")
    box_predict = mx.symbol.Variable(name="box_predict")
    ground_truth = mx.symbol.Variable(name="ground_truth")

    # roi warping
    roi_warping = mx.symbol.ROIWarping(*[relu_feature, box_predict, ground_truth], warped_shape=(28, 28),
                                       spatial_scale=0.5, name="roi_warping")
    roi_warping_pool = mx.symbol.Pooling(
            data=roi_warping, pool_type="max", kernel=(4, 4), stride=(4, 4), name="roi_warping_pool"
    )
    roi_warping_flatten = mx.symbol.Flatten(data=roi_warping_pool)

    roi_warping_fc1 = mx.symbol.FullyConnected(data=roi_warping_flatten, num_hidden=1024, name="roi_warping_fc1")
    roi_warping_bn1 = mx.symbol.BatchNorm(data=roi_warping_fc1, name="roi_warping_bn1")
    roi_warping_relu1 = mx.symbol.Activation(data=roi_warping_bn1, act_type="relu", name="roi_warping_relu1")

    roi_warping_fc2 = mx.symbol.FullyConnected(data=roi_warping_relu1, num_hidden=1024, name="roi_warping_fc2")
    roi_warping_bn2 = mx.symbol.BatchNorm(data=roi_warping_fc2, name="roi_warping_bn2")
    roi_warping_relu2 = mx.symbol.Activation(data=roi_warping_bn2, act_type="relu", name="roi_warping_relu2")

    # bbox prediction
    offset_predict = mx.symbol.FullyConnected(data=roi_warping_relu2, num_hidden=5, name="offset_predict")

    loss_all = mx.symbol.Group([offset_predict])

    return loss_all
