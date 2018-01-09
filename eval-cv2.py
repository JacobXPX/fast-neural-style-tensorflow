# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import img_preprocessing
import reader
import model
import time
import os
import cv2


tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_integer("port", 0, "camera device port")
FLAGS = tf.app.flags.FLAGS

# Initialize video capture
video_size = 1000
video_capture = cv2.VideoCapture(FLAGS.port)
video_capture.set(cv2.CAP_PROP_EXPOSURE,0.1)


def main(_):

    with tf.Graph().as_default():

        with tf.Session().as_default() as sess:
            # create placeholder for image streaming
            image_place = tf.placeholder(tf.float32, shape=[None,None,3])

            # Add batch dimension
            generated = model.net(tf.expand_dims(image_place, 0), training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            # Start video
            while True:
                # Generate frame
                frame = img_preprocessing.get_frame(video_capture, FLAGS.image_size)
                frame = sess.run(generated, feed_dict = {image_place:frame})

                # Resize window
                cv2.namedWindow("Style Transfer", 0)
                cv2.resizeWindow('Style Transfer', video_size,int(video_size*0.7))
                cv2.imshow('Style Transfer', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

