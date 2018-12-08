
"""Estimator wrapper for ResNet Keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models import build_resnet_model

def build_resnet_estimator(input_shape, n_classes,
                           depth=18, model_dir=None):
    """Build a TF Estimator instance for ResNet.
    Args:
        input_shape: (width, height, channel) of input images.
        n_classes: Number of output classes.
        depth: Depth of the model. Expected values are 18, 34, 50, 101 or 152.
        model_dir: Directory to store parameters.
    Returns:
        ResNet TF Estimator.
    """

    model = build_resnet_model(input_shape, n_classes, 
                               depth=depth)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model.compile(optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir=model_dir,
    )
    return estimator

