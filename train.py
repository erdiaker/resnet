
"""Train and evaluate a ResNet estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import tensorflow as tf

from estimators import build_resnet_estimator
from data import make_input_fn

def main(args):
    estimator = build_resnet_estimator(input_shape=tuple(args.img_shape),
                                       n_classes=args.n_labels,
                                       depth=args.depth,
                                       model_dir=args.model_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(
            args.train_tar_path, args.train_annotation_path,
            image_output_shape=tuple(args.img_shape),
            n_classes=args.n_labels,
            vec_encoder_path=args.vec_encoder_path,
        ),
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(
            args.valid_tar_path, args.valid_annotation_path,
            image_output_shape=tuple(args.img_shape),
            n_classes=args.n_labels,
            vec_encoder_path=args.vec_encoder_path,
        ),
        throttle_secs=600,
        start_delay_secs=600,
    )
    
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec,
    )
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Train and evaluate a ResNet estimator.')
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
    parser.add_argument('--train_tar_path',
                        default='data/train_0.tar.gz',
                        help='Path of OpenImagesV4 training images.')
    parser.add_argument('--train_annotation_path',
                        default='data/train-annotations-human-imagelabels-boxable.csv',
                        help='Path of OpenImagesV4 validation annotations.')
    parser.add_argument('--valid_tar_path', 
                        default='data/validation.tar.gz',
                        help='Path of OpenImagesV4 validation images.')
    parser.add_argument('--valid_annotation_path',
                        default='data/validation-annotations-human-imagelabels-boxable.csv',
                        help='Path of OpenImagesV4 training annotations.')
    args = parser.parse_args()

    main(args)

