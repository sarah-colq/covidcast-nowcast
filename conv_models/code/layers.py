import six
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.keras.layers import Conv1D
import functools


class CustomConv1D(Conv1D):

    def __init__(self, filters,  filter_bank, activation=None, **kwargs):
        super(CustomConv1D, self).__init__(filters, kernel_size=1,
                                           activation=None, use_bias=False, **kwargs)
        self.stacked_filter_bank = tf.expand_dims(
            tf.expand_dims(tf.stack(filter_bank), axis=-1), axis=-1)
        self.stacked_filter_bank = tf.cast(
            self.stacked_filter_bank, dtype=self.dtype)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))

        kernel_shape = (self.stacked_filter_bank.shape[0], 1, 1, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        self.bias = None

        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

    def call(self, inputs):
        kernel_ = tf.reduce_sum(self.kernel * self.stacked_filter_bank, axis=0)
        outputs = self._convolution_op(inputs, kernel_)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class CustomConvGamma1D(Conv1D):

    def __init__(self, p, filters=2, activation=None, **kwargs):
        super(CustomConvGamma1D, self).__init__(filters, kernel_size=1,
                                           activation=None, use_bias=False, **kwargs)
        self._kernel = tf.linspace(p,1,p)
        self._kernel = tf.reshape(self._kernel, shape=(p,1,1))
        self._kernel = tf.cast(
            self._kernel, dtype=self.dtype)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))

        kernel_shape = (1, 1, 1, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        self.bias = None

        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

    def call(self, inputs):
        alpha = self.kernel[0,0,0,0]
        beta = self.kernel[0,0,0,1]

        exp_beta = tf.math.exp(-beta*self._kernel)
        x_alpha = self._kernel ** (alpha-1)
        self.kernel_ = x_alpha * exp_beta
        
        outputs = self._convolution_op(inputs, self.kernel_)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs