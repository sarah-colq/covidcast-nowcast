import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Lambda, Concatenate
from layers import CustomConv1D


class Model(tf.keras.Model):
    """
    Simple convolutional model for fitting leading indicators to case counts
    Attributes:
        p: Size of the pd kernel
        m (int): Number of geo_values
    """

    def __init__(self, p=30, m=1, filter_bank=[], kernel_constraint=None, kernel_regularizer=None):
        super(Model, self).__init__()
        """
        Args: 
            p (int): Size of the p_d kernel. Value ignored when filter_bank given. 
            m (int): Number of geo_values
            filter_bank: A list of kernels. The kernels should be
                1-dimensional arrays each with the correct orientation for
                cross-correlation and length.
        """
        assert p > 0 and isinstance(
            p, int), "p must be an integer greater than 0"

        self.p = p
        self.m = m
        self.kernel_constraint = kernel_constraint
        self.kernel_regularizer = kernel_regularizer
        self.conv_layers = []

        if filter_bank:
            for i in range(m):
                layer = CustomConv1D(
                    filters=1,
                    filter_bank=filter_bank,
                    kernel_constraint=kernel_constraint,
                    kernel_regularizer=kernel_regularizer,
                    name='custom_conv{}'.format(i),
                )
                self.conv_layers.append(layer)
        else:
            for i in range(m):
                layer = Conv1D(
                    filters=1,
                    kernel_size=p,
                    use_bias=False,
                    kernel_constraint=kernel_constraint,
                    kernel_regularizer=kernel_regularizer,
                    name='conv{}'.format(i),
                )
                self.conv_layers.append(layer)

    def call(self, x):
        if self.m == 1:
            return self.conv_layers[0](x)
        else:
            split_inputs = tf.split(x, num_or_size_splits=self.m, axis=-1)
            split_outputs = [self.conv_layers[i](
                split_inputs[i]) for i in range(self.m)]
            joined_outputs = Concatenate()(split_outputs)
            return joined_outputs

    def train_step(self, inputs):
        X, Y = inputs

        assert X.shape[1] == Y.shape[1], "Size of X and Y should be the same shape but found, {} vs {}".format(
            X.shape[1], Y.shape[1])
        X_padded = tf.pad(
            X,
            paddings=[[0, 0], [self.p-1, 0], [0, 0]],
        )

        with tf.GradientTape() as tape:
            Y_hat = self(X_padded)
            loss = self.loss(Y, Y_hat)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(Y, Y_hat)
        return {m.name: m.result() for m in self.metrics}
