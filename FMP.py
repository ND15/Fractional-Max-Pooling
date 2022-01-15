import tensorflow as tf
from tensorflow import keras
from keras.layers import *


class FractionalMaxPooling2D(Layer):
    def __init__(self, pool_ratio=None, pseudo_random=True, overlap=False, **kwargs):
        self.pool_ratio = pool_ratio,
        self.pseudo_random = pseudo_random,
        self.overlapping = overlap

        self.fmp_config = {
            "pool_ratio": pool_ratio,
            "pseudo_random": pseudo_random,
            "overlapping": overlap
        }
        super(FractionalMaxPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, input):
        [ret_val, rows, cols] = tf.nn.fractional_max_pool(input, pooling_ratio=(1.0, 1.25, 1.25, 1.0),
                                                          pseudo_random=True, overlapping=self.overlapping, seed=1)
        return ret_val

    def compute_output_shape(self, input_shape):
        if (input_shape[0] != None):
            batch_size = (int(input_shape[0] / self.pool_ratio[0]))

        else:
            batch_size = input_shape[0]
        channels = int(input_shape[1] / self.pool_ratio[1])
        width = int(input_shape[2] / self.pool_ratio[2])
        height = int(input_shape[3] / self.pool_ratio[3])
        return (batch_size, channels, width, height)

    def get_config(self):
        base_config = super(FractionalMaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(self.fmp_config.items()))
