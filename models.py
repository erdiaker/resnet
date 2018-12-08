
"""Building blocks and models for deep residual networks.

Original paper:
    He et al., 2015
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

def _build_regular_block(input_layer, filters=64, kernel_size=(3, 3), downsample=False, 
                         **kwargs):
    """Building block used in shallower ResNets (i.e., 18-34 layers)."""

    initial_stride = (2, 2) if downsample else (1, 1)

    # conv layer 1
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=initial_stride, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # conv layer 2
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # shortcut layer
    if not downsample:
        x = Add()([x, input_layer])
    else:
        projected_input = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=initial_stride, 
                                  padding='same')(input_layer)
        x = Add()([x, projected_input])

    return x

def _build_bottleneck_block(input_layer, filters=64, kernel_size=(3, 3), downsample=False,
                            initial_block=False, **kwargs):
    """Building block used in deeper ResNets (i.e., >=50 layers)."""

    initial_stride = (2, 2) if downsample else (1, 1)

    # conv layer 1
    x = Conv2D(filters=filters, kernel_size=(1, 1),
               strides=initial_stride, padding='same')(input_layer) 
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # conv layer 2
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # conv layer 3
    x = Conv2D(filters=filters * 4, kernel_size=(1, 1),
               strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # shortcut layer
    if not downsample and not initial_block:
        x = Add()([x, input_layer])
    else:
        projected_input = Conv2D(filters=filters * 4, kernel_size=(1, 1),
                                 strides=initial_stride, 
                                 padding='same')(input_layer)
        x = Add()([x, projected_input])

    return x

def _stack_blocks(input_layer, bottleneck=False, 
                 reps=(2, 2, 2, 2), filters=(64, 128, 256, 512), 
                 kernel_size=(3, 3),
                 conv1_kernel_size=(7, 7), conv1_strides=(2, 2), 
                 conv1_filters=64,
                 pool1_size=(3, 3), pool1_strides=(2, 2),
                 n_classes=1000,
                 **kwargs):
    """Stack building/bottleneck blocks.
    
    Args:
        input_layer: input layer, an instance of tf.keras.layers.Input()
        bottleneck: if True, bottleneck blocks are used (suitable for deeper nets)
        reps: number of repetition for each block (tuple of 4 ints) 
        filters: number of filters for each block (tuple of 4 ints)
        kernel_size: convolution window size in blocks
        conv1_kernel_size: window size of the initial conv layer
        conv1_strides: strides of the initial conv layer
        pool1_kernel_size: windows size of the initial pooling layer
        pool1_strides: strides of the initial pooling layer
        n_classes: number of output classes
    Returns:
        ResNet model without the input layer.
    """
    assert len(reps) == len(filters) == 4

    # initial convolution
    x = Conv2D(filters=conv1_filters, kernel_size=conv1_kernel_size, 
               strides=conv1_strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=pool1_size, 
                     strides=pool1_strides, padding='same')(x)

    # resnet blocks
    build_block = _build_bottleneck_block if bottleneck else _build_regular_block
    for stack_type in range(4):
        for rep in range(reps[stack_type]):
            x = build_block(x, 
                            filters=filters[stack_type], 
                            kernel_size=kernel_size, 
                            downsample=(stack_type != 0),
                            initial_block=(rep == 0))

    # fully connected
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax')(x)

    return x

def _get_config(depth):
    """Get hyperparameters for given model depth.
    
    Args:
        depth: One of 18, 34, 50, 101, or 152. 
    Returns:
        Dictionary of hyperparameters.
    """
    if depth == 18:
        config = dict(reps=(2, 2, 2, 2), 
                      bottleneck=False)
    elif depth == 34:
        config = dict(reps=(3, 4, 6, 3),
                      bottleneck=False)
    elif depth == 50:
        config = dict(reps=(3, 4, 6, 3),
                      bottleneck=True)
    elif depth == 101:
        config = dict(reps=(3, 4, 23, 3),
                      bottleneck=True)
    elif depth == 152:
        config = dict(reps=(3, 8, 36, 3),
                      bottleneck=True)
    else:
        raise ValueError("Depth must be 18, 34, 50, 101 or 152.")
    return config

def build_resnet_model(input_shape, n_classes, depth=18):
    """Build a ResNet Keras model
    Args:
        input_shape: (width, height, channel) of input images.
        n_classes: Number of output classes.
        depth: Depth of the model. Expected values are 18, 34, 50, 101 or 152.
    Returns:
        ResNet Keras model
    """
    config = _get_config(depth)

    inputs = Input(shape=input_shape)
    outputs = _stack_blocks(inputs, **config)
    model = Model(inputs=inputs, outputs=outputs)
    return model

