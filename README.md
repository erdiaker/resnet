
ResNet with Tensorflow High-Level APIs
======================================

An implementation of [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) using Tensorflow high-level APIs.

## Data
Scripts to train and evaluate the model with [Open Images V4 dataset](https://storage.googleapis.com/openimages/web/index.html) are included in `data.py`. Instructions to download the dataset are [here](https://github.com/cvdfoundation/open-images-dataset).

## Training and Evaluation
```sh
python3 train.py \
    --train_tar_path='train.tar.gz' \ 
    --train_annotation_path='train-annotations-human-imagelabels-boxable.csv' \
    --valid_tar_path='validation.tar.gz' \
    --valid_annotation_path='validation-annotations-human-imagelabels-boxable.csv' \
    --depth=50 \
    --model_dir='./logs'
```
Type `python3 train.py --help` to see the full list of arguments.

## Prediction
```sh
python3 predict.py \
    --test_tar_path='test.tar.gz' \ 
    --depth=50 \
    --model_dir='./logs' \
    --output_type='labels'
```
Type `python3 predict.py --help` to see the full list of arguments.

