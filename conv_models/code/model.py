import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class Model(tf.keras.Model):
    """
    Simple convolutional model for fitting leading indicators to case counts
    Attributes:
        p: Size of the pd kernel
        m (int): Number of geo_values
        p_conv: Convolutional layer for p_d kernel
    """

    def __init__(self, p=30,m=1, kernel_constraint=None):
        super(Model, self).__init__()
        """
        Args:
            p (int): Size of the p_d kernel
            m (int): Number of geo_values
            p_conv: Convolutional layer for p_d kernel
        """
        assert p > 0 and isinstance(
            p, int), "p must be an integer greater than 0"

        self.p = p
        self.m = m
        self.p_conv = Conv2D(filters=m, kernel_size=(m,p), kernel_constraint=kernel_constraint, use_bias=False)

    def call(self, x):
        return self.p_conv(x)

    def train_step(self, inputs):
        X, Y = inputs
        assert X.shape[1] == Y.shape[1], "Size of X and Y should be the same shape but found, {} vs {}".format(
            X.shape[1], Y.shape[1])
        X_padded = tf.pad(
            X,
            paddings =[[0, 0], [0, 0], [self.p-1, 0], [0, 0]],
        )

        with tf.GradientTape() as tape:
            Y_hat = self.p_conv(X_padded, training=True)
            loss = self.loss(Y, Y_hat)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(Y, Y_hat)
        return {m.name: m.result() for m in self.metrics}