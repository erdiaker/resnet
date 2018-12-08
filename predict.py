
"""Predict image labels with given ResNet estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import pandas as pd
import tensorflow as tf

from estimators import build_resnet_estimator
from data import make_input_fn
from data import load_label_encoder

def main(args):
    estimator = build_resnet_estimator(input_shape=tuple(args.img_shape),
                                       n_classes=args.n_labels,
                                       depth=args.depth,
                                       model_dir=args.model_dir)

    pred_iter = estimator.predict(
        input_fn=make_input_fn(
            args.test_tar_path,
            image_output_shape=tuple(args.img_shape),
            n_classes=args.n_labels,
        ),
    )
    
    encoder = load_label_encoder(args.vec_encoder_path)

    for pred in pred_iter:
        probs = pred['dense']
        if args.output_type == 'probs':
            print(probs)
        else:
            print(encoder.inverse_transform(probs, args.output_threshold))
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Predict image labels.')
    parser.add_argument('--model_dir',
                        default='./logs',
                        help='Directory to save checkpoints.')
    parser.add_argument('--img_shape',
                        default=(224, 224, 3), nargs=3, type=int,
                        help='3-tuple shape of images.')
    parser.add_argument('--n_labels',
                        default=1000, type=int,
                        help='Number of unique labels.')
    parser.add_argument('--depth',
                        choices=[18, 34, 50, 101, 152], default=50, type=int,
                        help='Depth of the model. 18, 34, 50, 101, or 152.')
    parser.add_argument('--vec_encoder_path',
                        default='./logs/vec_encoder.pickle',
                        help='Path to serialize/deserialize the encoder for labels.')
    parser.add_argument('--test_tar_path',
                        default='data/test.tar.gz',
                        help='Path of OpenImagesV4 test images.')
    parser.add_argument('--output_type',
                        choices=['probs', 'labels'],
                        default='labels',
                        help='Choice for output. Probabilities, or labels.')
    parser.add_argument('--output_threshold',
                        default=0.2, type=float,
                        help='Probability threshold for labels to output.')
    args = parser.parse_args()

    main(args)


