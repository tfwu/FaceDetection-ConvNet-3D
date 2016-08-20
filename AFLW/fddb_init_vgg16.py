import mxnet as mx
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_from_vgg16(ctx, rpn_symbol, vgg16fc_args, vgg16fc_auxs):
    fc_args = vgg16fc_args.copy()
    fc_auxs = vgg16fc_auxs.copy()

    for k, v in fc_args.items():
        if v.context != ctx:
            fc_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fc_args[k])

    for k, v in fc_auxs.items():
        if v.context != ctx:
            fc_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fc_auxs[k])

    return fc_args, fc_auxs
