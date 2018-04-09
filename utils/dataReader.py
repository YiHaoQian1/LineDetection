import os
import numpy as np
import tensorflow as tf
from PIL import Image

"""
    Using resnet semantic segmentation pretrained model, this model is trained in bgr, so we need to change the order of color channel
"""


label_colors = [[0,0,0]
                # 0 = background
                ,[125,125,125]
                # 1 = line
                ]

def color_labels(mask,num_classes = 1):
    n,h,w,c = mash.shape
    outputs = np.zeros((n,h,w,3),dtype=np.uint8)
    for i in range(n):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
                else:
                    printf("wrong index")
        outputs[i] = np.array(img)
    return outputs


def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    """Resize label and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch







