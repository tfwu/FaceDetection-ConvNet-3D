import mxnet as mx


def get_vgg16_finetune():
    ell_label = mx.symbol.Variable(name="ell_label")
    bbox_weight = mx.symbol.Variable(name="bbox_weight")
    data = mx.symbol.Variable(name="data")

    roi_warping_fc1 = mx.symbol.FullyConnected(data=data, num_hidden=1024, name="roi_warping_fc1")
    roi_warping_relu1 = mx.symbol.Activation(data=roi_warping_fc1, act_type="relu", name="roi_warping_relu1")
    roi_warping_dropout = mx.symbol.Dropout(data=roi_warping_relu1, p=0.5, name="roi_warping_dropout1")

    roi_warping_fc2 = mx.symbol.FullyConnected(data=roi_warping_dropout, num_hidden=1024, name="roi_warping_fc2")
    roi_warping_relu2 = mx.symbol.Activation(data=roi_warping_fc2, act_type="relu", name="roi_warping_relu2")
    roi_warping_dropout2 = mx.symbol.Dropout(data=roi_warping_relu2, p=0.5, name="roi_warping_dropout2")

    # bbox prediction
    offset_predict = mx.symbol.FullyConnected(data=roi_warping_dropout2, num_hidden=6, name="offset_predict")
    ellipse_predict_loss = mx.symbol.SmoothL1Loss(
            *[offset_predict, bbox_weight, ell_label], name="ellipse_predict_loss"
    )

    loss_all = mx.symbol.Group([ellipse_predict_loss])
    return loss_all
