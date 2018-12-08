
"""Input functions for OpenImages V4 dataset.

Dataset description: https://storage.googleapis.com/openimages/web/index.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tarfile
import pickle

from io import BytesIO
from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array 

class OpenImagesDataGen(object):
    '''Generator for Open Images V4.'''

    def __init__(self, tar_path, 
                 img_resize_shape=None,
                 include_annotations=False,
                 annotation_path=None,
                 n_classes=None,
                 vec_encoder_path=None):
        """Initialize generator.

        Args:
            tar_path: Path of the tar file containing images.
            img_resize_shape: width and height of the images to resize.
            include_annotations: When True, labels are included in the output.
            annotation_path: Path of the .csv file containing image annotations.
            n_classes: Number of unique image labels.
            vec_encoder_path: Path to serialize/deserialize the encoder for labels.
                If set, deserialization will be tried at first.
        Returns:
            None
        """
        super(OpenImagesDataGen, self).__init__()

        self._tar = tarfile.open(tar_path)

        self._annodf = None
        self._vec_encoder = None
        if annotation_path is not None:
            self._annodf = pd.read_csv(annotation_path)
            try:
                with open(vec_encoder_path, 'rb') as f:
                    self._vec_encoder = pickle.load(f)
            except:
                labels = self._annodf['LabelName'].unique()
                self._vec_encoder = OpenImagesVectorEncoder(labels, n_classes)
                if vec_encoder_path is not None:
                    os.makedirs(os.path.dirname(vec_encoder_path), exist_ok=True)
                    with open(vec_encoder_path, 'wb+') as f:
                        pickle.dump(self._vec_encoder, f)

        self._img_resize_shape = img_resize_shape
        self._include_annotations = include_annotations

        self._members = self._get_tar_members(self._tar)
        self._idx = 0

    def __len__(self):
        return len(self._members)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration

        val = self._get_item(self._idx)       
        self._idx += 1 
        return val

    def __call__(self):
        return self

    def _get_item(self, idx):
        batch_x = self._get_x(idx)
        if not self._include_annotations:
            return batch_x
        else:
            batch_y = self._get_y(idx) 
            return batch_x, batch_y

    def _get_x(self, idx):
        img_path = self._members[idx].name
        img_arr = self._load_image(img_path, self._img_resize_shape)
        return img_arr

    def _get_y(self, idx):
        img_path = self._members[idx].name
        img_id = re.findall(r'[\/]([A-Za-z0-9]+)\.jpg', img_path)[0]
        rows = self._annodf[self._annodf['ImageID'] == img_id]
        labels = rows['LabelName'].values
        encoded = self._vec_encoder.transform(labels)
        return encoded 
    
    def _load_image(self, img_path, shape=None):
        img = None

        with self._tar.extractfile(img_path) as f:
            img = Image.open(BytesIO(f.read()))

        if shape is not None:
            img = img.resize(shape)

        img_arr = img_to_array(img)
        return img_arr

    def _get_tar_members(self, tar):
        members = tar.getmembers()
        members = list(filter(lambda x: re.match(r'.*[\/][A-Za-z0-9]+\.jpg', x.name), 
                              members))
        return members


class OpenImagesVectorEncoder(object):
    '''Vector encoder for Open Images V4 labels.'''

    def __init__(self, labels, n_classes):
        '''Initialize encoder.

        Args:
            labels: A list of unique labels.
        Returns:
            None
        '''
        assert(n_classes >= len(labels))

        self._labels = labels
        self._ndim = n_classes 
        self._label2index = {label: i for i, label in enumerate(labels)} 

    def transform(self, labels):
        vec = np.zeros((self._ndim))
        for label in labels:
            vec[self._label2index[label]] = 1
        return vec

    def inverse_transform(self, vec, threshold=0.5):
        labels = []
        for i, elt in enumerate(vec):
            if elt > threshold:
                i = min(i, len(self._labels) - 1)
                labels.append(self._labels[i])
        return labels

def load_label_encoder(path):
    """Load previously serialized label encoder. 

    Typically used to obtain original labels while predicting.

    Args:
        path: Path of the serialized encoder.
    Returns:
        Label encoder
    """
    encoder = None
    with open(path, 'rb') as f:
        encoder = pickle.load(f)   
    return encoder

def make_input_fn(tar_path, annotation_path=None, 
                  image_output_shape=(224, 224, 3),
                  n_classes=1000,
                  batch_size=32, repeat=1,
                  vec_encoder_path=None):

    """Create an input function to use with an estimator.
    Args:
        tar_path: Path of the tar file containing images.
        annotation_path: Path of the .csv file containing image annotations.
            Set the value to "None" to omit label outputs while predicting.
        img_output_shape: (width, height, channel) of generated images.
        n_classes: Number of unique image labels.
        batch_size: Number of samples in each batch.
        repeat: Number of epochs.
        vec_encoder_path: Path to serialize/deserialize the vector encoder 
            used for labels. Set to None for no serialization.
    Returns:
        Input function returning a tf.dataset.Dataset instance.
    """
    def input_fn():
        dgen = OpenImagesDataGen(
            tar_path=tar_path,
            img_resize_shape=image_output_shape[:2],
            include_annotations=(annotation_path is not None),
            annotation_path=annotation_path,
            n_classes=n_classes,
            vec_encoder_path=vec_encoder_path,
        )
        if annotation_path is not None:
            dataset = tf.data.Dataset.from_generator(dgen, 
                output_types=(tf.float32, tf.int32),
                output_shapes=(image_output_shape, n_classes)
            )
        else:
            dataset = tf.data.Dataset.from_generator(dgen, 
                output_types=(tf.float32),
                output_shapes=(image_output_shape)
            )

        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(repeat)
        return dataset
    return input_fn
    
