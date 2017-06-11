# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
from fast_rcnn.config import cfg
from PIL import ImageEnhance
from PIL import Image

def normalize_image(im):
    """Change in range [-1,1]"""
    im *= (2.0/255.0)
    im -= 1
    return im

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size,perturb=None,normalize=False):
    """Mean subtract and scale an image for use in a blob."""
    if cfg.TRAIN.USE_RANDOM_ENHANCEMENT :
        # TODO : see same operations in opencv
        # convert to RGB
        imobj = Image.fromarray(im[:,:,[2,1,0]])
        enhancers = [ImageEnhance.Color, ImageEnhance.Brightness, ImageEnhance.Contrast]
        enhance_value = np.random.rand(len(enhancers)) + .5 
        enhancer_order = np.random.permutation(np.arange(len(enhancers)))
        for i in xrange(len(enhancers)):
            enhancer = enhancers[i](imobj)
            imobj = enhancer.enhance(enhance_value[enhancer_order[i]])
        im = np.asarray(imobj)
        # convert back to BGR
        im = im[:,:,[2,1,0]]
        # add lighting noise
        im = im.astype(np.float32, copy=False)
        im += perturb
        im[im>255] = 255
        im[im<0] = 0    

    im = im.astype(np.float32, copy=False)
    if normalize :
        im = normalize_image(im)
    else :
        im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
