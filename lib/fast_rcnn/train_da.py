# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
from train import SolverWrapper,filter_roidb
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os,cv2

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class DASolverWrapper(SolverWrapper):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over the snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """
    def __init__(self, solver_prototxt, src_roidb,target_roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(src_roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            if type(pretrained_model) == type([]):
                for p_model in pretrained_model:
                    print ('Loading pretrained model '
                        'weights from {:s}').format(p_model)
                    self.solver.net.copy_from(p_model)
            else :
                print ('Loading pretrained model '
                    'weights from {:s}').format(pretrained_model)
                self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)
        
        
        self.solver.net.layers[0].set_roidb(src_roidb)
        if cfg.TRAIN.ADAPTATION_LOSS in ['DC_LOSS', 'MMD_LOSS', 'CORAL_LOSS']:
            assert target_roidb , "target_roidb not initialized"
            self.solver.net.layers[4].set_roidb(target_roidb)
    
def train_net(solver_prototxt, src_roidb, target_roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    src_roidb = filter_roidb(src_roidb)
    # target_roidb = filter_roidb(target_roidb)
    sw = DASolverWrapper(solver_prototxt, src_roidb,target_roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
