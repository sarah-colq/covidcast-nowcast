import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Lambda, Concatenate
from layers import CustomConv1D, CustomConvGamma1D


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
        self.lam = lam  # (NEW) used for the regularization part
        self.conv_layers = []

        if filter_bank:
            # (NEW) used for the regularization part
            self.uses_filter_bank = True
            for i in range(m):
                layer = CustomConv1D(
                    filters=1,
                    filter_bank=filter_bank,
                    kernel_constraint=kernel_constraint,
                    kernel_regularizer=kernel_regularizer,
                    name='customConv{}'.format(i),
                )
                self.conv_layers.append(layer)
        else:
            for i in range(m):
                # (NEW) used for the regularization part
                self.uses_filter_bank = False
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
                    # shape = (num_filters, length_of_each_filter)
                    A_T = self.conv_layers[i].kernel * \
                        self.conv_layers[i].stacked_filter_bank
                    A_T = tf.squeeze(A_T)
                    # shape = (length_of_each_filter, num_filters)
                    A = tf.transpose(A_T, perm=[1, 0])
                    # shape = (num_filters, num_filters), this is effectively the Gram matrix of the set of filters [alpha_1*z_1, ..., alpha_i*z_i, ..., alpha_m*z_m]
                    K = tf.linalg.matmul(A_T, A)
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


class ModelGamma(tf.keras.Model):
    """
    Simple convolutional model for fitting leading indicators to case counts
    Attributes:
        p: Size of the pd kernel
        m (int): Number of geo_values
    """

    def __init__(self, p=30, m=1, kernel_constraint=None, kernel_regularizer=None, lam=1):
        """
        Args: 
            p (int): Size of the p_d kernel. Value ignored when filter_bank given. 
            m (int): Number of geo_values
            lam (float): weight for log barrier penalty, helps to enforce the gamma parameters are positive
        """
        super(ModelGamma, self).__init__()
        assert p > 0 and isinstance(
            p, int), "p must be an integer greater than 0"

        self.p = p
        self.m = m
        self.kernel_constraint = kernel_constraint
        self.kernel_regularizer = kernel_regularizer
        self.lam = lam
        self.conv_layers = []

        for i in range(m):
            layer = CustomConvGamma1D(
                p=p,
                kernel_constraint=kernel_constraint,
                kernel_initializer=tf.keras.initializers.Constant(value=2),
                kernel_regularizer=kernel_regularizer,
                name='customConvGamma{}'.format(i),
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
            for i in range(self.m):
                loss -= self.lam * tf.reduce_sum(tf.math.log(self.conv_layers[i].kernel))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        for var in self.trainable_variables:
            var.assign(tf.math.maximum(0., var))

        self.compiled_metrics.update_state(Y, Y_hat)
        return {m.name: m.result() for m in self.metrics}
