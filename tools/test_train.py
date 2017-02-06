#!/usr/bin/env python

"""Test a new model on test data set when training."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys, glob

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net-prefix', dest='net_prefix',
                        help='models path prefix to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='coca_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if args.cpu_mode:
        caffe.set_mode_cpu()
        cfg.USE_GPU_NMS = False
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    net_pattern = args.net_prefix + '*.caffemodel'
    log_file = args.net_prefix + '_test_train.log'
    with open(log_file, 'w'):
        pass
    prev_models = []
    while True:
        models = glob.glob(net_pattern)
        new_models = [model for model in models if model not in prev_models]
        prev_models = models
        if new_models:
            for new_model in new_models:
                net = caffe.Net(args.prototxt, new_model, caffe.TEST)
                net.name = os.path.splitext(os.path.basename(new_model))[0]
                m_ap = test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis)
                if m_ap is not None:
                    with open(log_file, 'a') as log:
                        log.write('Model: {}, mAP: {}\n'.format(new_model, m_ap))
        else:
            print 'Waiting for new models...'
            time.sleep(10)
