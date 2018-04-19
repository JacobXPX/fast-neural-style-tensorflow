from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2

def _smallest_size_at_least(height, width, target_height, target_width):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    target_height = tf.convert_to_tensor(target_height, dtype=tf.int32)
    target_width = tf.convert_to_tensor(target_width, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    target_height = tf.to_float(target_height)
    target_width = tf.to_float(target_width)

    scale = tf.cond(tf.greater(target_height / height, target_width / width),
                    lambda: target_height / height,
                    lambda: target_width / width)
    new_height = tf.to_int32(tf.round(height * scale))
    new_width = tf.to_int32(tf.round(width * scale))
    return new_height, new_width


def _img_resize(image, target_height, target_width):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image .
      height: an int32 scalar indicating the target height.
      width: an int32 scalar indicating the target width.

    Returns:
      resized_image: A 3-D resized image.
    """
    target_height = tf.convert_to_tensor(target_height, dtype=tf.int32)
    target_width = tf.convert_to_tensor(target_width, dtype=tf.int32)
    shape = tf.shape(image)

    new_height, new_width = _smallest_size_at_least(shape[0], shape[1], 
				target_height, target_width)
    resized_image = cv2.resize(image, (new_width.eval(), new_height.eval()))
    return resized_image

def get_frame(video, image_size):
    """get the frame from video and preprocess on the frame.

    Args:
      video: video object from cv2.VideoCapture.
      shrink: the ratio to shrink the image

    Returns:
      resized_image: A 3-D resized image.
    """
    ret, frame = video.read()
    frame = _img_resize(frame, image_size, image_size)
    return frame


