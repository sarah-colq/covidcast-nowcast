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

    def __init__(self, p=30, m=1, kernel_constraint=None, kernel_regularizer=None, filter_bank=None, lam=0):
        """
        Args: 
            p (int): Size of the p_d kernel. Value ignored when filter_bank given. 
            m (int): Number of geo_values
            filter_bank: A list of kernels. The kernels should be
                1-dimensional arrays each with the correct orientation for
                cross-correlation and length.
        """
        super(Model, self).__init__()
        assert p > 0 and isinstance(
            p, int), "p must be an integer greater than 0"

        self.p = p
        self.m = m
        self.kernel_constraint = kernel_constraint
        self.kernel_regularizer = kernel_regularizer
        self.lam = lam
        self.conv_layers = []

        if filter_bank:
            self.uses_filter_bank = True # (NEW) used for the regularization part
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
                self.uses_filter_bank = False # (NEW) used for the regularization part
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

            #####################################
            # (NEW) Code for regularization
            #####################################
            m = len(self.conv_layers)
            n = self.conv_layers[0].trainable_variables[0].shape[0]

            if self.uses_filter_bank:
                loss_regularizer = 0

                for i in range(m):
                    # A_T is the set of filters stacked horizontally multiplied by their respective weights
                    A_T = self.conv_layers[i].kernel * self.conv_layers[i].stacked_filter_bank # shape = (num_filters, length_of_each_filter)
                    A_T = tf.squeeze(A_T)
                    A = tf.transpose(A_T, perm=[1,0]) # shape = (length_of_each_filter, num_filters)
                    K = tf.linalg.matmul(A_T,A) # shape = (num_filters, num_filters), this is effectively the Gram matrix of the set of filters [alpha_1*z_1, ..., alpha_i*z_i, ..., alpha_m*z_m]
                    L2 = tf.reduce_sum(K ** 2)
                    loss_regularizer += L2
                
                loss += self.lam*loss_regularizer
            
            #####################################

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        for var in self.trainable_variables:
            var.assign(tf.math.maximum(0., var))

        self.compiled_metrics.update_state(Y, Y_hat)
        return {m.name: m.result() for m in self.metrics}
