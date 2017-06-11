#!/usr/bin/env python

#--------------------------------------------------------
# Domain Adaptation
# Written By Debjeet Majumdar
#--------------------------------------------------------

import _init_paths
from fast_rcnn.train import get_training_roidb
from fast_rcnn.train_da import train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import os
import caffe
import argparse
import pprint
import numpy as np
import sys
from train_net import combined_roidb 

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--gan_weights', dest='pretrained_gan_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('-s','--srcimdb', dest='src_imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('-t','--tgtimdb', dest='target_imdb_name',
                        help='dataset to train on',
                        default='minicoco_2014_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def add_domain_label(roidb,domain):
    """
    Adds a domain label to all roidbs
        Source : 0
        Target : 1
    """
    for roi in roidb:
        roi['domain']=domain

def combined_target_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    PROPOSAL_METHOD = 'target'
    imdb.set_proposal_method(PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(PROPOSAL_METHOD)
    roidb = get_training_roidb(imdb)
    return imdb,roidb

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)
    
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    if not cfg.TRAIN.IS_ADAPTATION_NETWORK :
        print "Configuration for adaptation network not set, Please check."
        sys.exit(1) 

    if cfg.TRAIN.ADAPTATION_LOSS == 'DC_LOSS':

        target_imdb,target_roidb = combined_target_roidb(args.target_imdb_name)
        print '{:d} target roidb entries'.format(len(target_roidb))
        
        src_imdb, src_roidb = combined_roidb(args.src_imdb_name)
        print '{:d} source roidb entries'.format(len(src_roidb))

        add_domain_label(target_roidb,1)
        add_domain_label(src_roidb,0)
    
        output_dir = get_output_dir(target_imdb)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        train_net(args.solver, src_roidb, target_roidb, output_dir,
                    pretrained_model=args.pretrained_model,
                    max_iters=args.max_iters)

    elif cfg.TRAIN.ADAPTATION_LOSS == 'GAN_LOSS':
        assert args.pretrained_gan_model
        src_imdb, src_roidb = combined_roidb(args.src_imdb_name)
        print '{:d} source roidb entries'.format(len(src_roidb))

        output_dir = get_output_dir(src_imdb)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        train_net(args.solver, src_roidb,None, output_dir,
                    pretrained_model=[args.pretrained_model,args.pretrained_gan_model],
                    max_iters=args.max_iters)
        # train_net(args.solver, src_roidb,None, output_dir,
        #             pretrained_model=args.pretrained_model,
        #             max_iters=args.max_iters)
        # train_net(args.solver, src_roidb,None, output_dir,
        #             pretrained_model=args.pretrained_gan_model,
        #             max_iters=args.max_iters)
    else :
        target_imdb,target_roidb = combined_target_roidb(args.target_imdb_name)
        print '{:d} target roidb entries'.format(len(target_roidb))
        
        src_imdb, src_roidb = combined_roidb(args.src_imdb_name)
        print '{:d} source roidb entries'.format(len(src_roidb))

        output_dir = get_output_dir(target_imdb)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        train_net(args.solver, src_roidb, target_roidb, output_dir,
                    pretrained_model=args.pretrained_model,
                    max_iters=args.max_iters)
