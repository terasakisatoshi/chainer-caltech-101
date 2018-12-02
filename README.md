# chainer-caltech-101

This is an example of image classification with Chainer using [caltech-101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)

you can try it with Chainer version 4.5.0, 5.0.0 and even 6.0.0a.

# Usage
Do the following procedures (See also my documents on Qiita).

- https://qiita.com/SatoshiTerasaki/items/c0a5a25b8bb82e95371b

## Download original dataset

Go to http://www.vision.caltech.edu/Image_Datasets/Caltech101/ and Click `Collection of pictures: 101_ObjectCategories.tar.gz (131Mbytes)` or use `wget` from terminal

```console

$ wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
```

## Reshape image data

```console
$ python reshape.py --source_dir 101_ObjectCategories --target_dir reshaped
```

You will wait for few seconds.

## Create dataset file

```console
$ python create_dataset.py reshaped
```

It will create `train.txt, test.txt, label.txt`

## Calculate mean

```console
$ python compute_mean.py train.txt
```

## Train GoogLeNet

```console
$ python train.py train.txt test.txt -a googlenet -j 8 -g 0
```

## Evaluate your trained model

```
python predict.py
```

If you install `ideep4py` you can accelerate  the inference speed with optional argument `--ideep`.

```
python predict.py --ideep
```

I uploaded output files `train.txt, test.txt, label.txt mean.npy` and pretraind model of GoogLeNet in `./example`.
You can use it by doing the following command.
```
python predict.py --trained ./example
```

# ChainerX

Chainer Team has released new backend called `ChainerX` its learning speed is crazy fast for MNIST.

## How to install ChainerX

Go to official Chainer repository. You can access from README.md

or read my post of Qiita (written in Japanese) https://qiita.com/SatoshiTerasaki/items/defbb1ea49b88c452118

## train ResNet50

They are also prepare training script of ImageNet (base net is ResNet50).
I take these scripts from https://github.com/chainer/chainer/blob/master/chainerx_cc/examples/imagenet/train_imagenet.py.
And then what to do is...

```
python train_chx_resnet.py train.txt test.txt -j 8 -d cuda:0
```

## predict

For now (2018/12/02) It seems ChainerX team do not release feature of save model. like `chainer.serializers.save_npz`.
But I think you can do it.

# References

- L. Fei-Fei, R. Fergus and P. Perona. Learning generative visual models
from few training examples: an incremental Bayesian approach tested on
101 object categories. IEEE. CVPR 2004, Workshop on Generative-Model
Based Vision. 2004
- training script is taken from chainer repository
  - https://github.com/chainer/chainer/tree/master/examples/imagenet
  - licensed under MIT
- reshape script or other utils to begin training is taken from `chainer_imagenet_tools`
  - https://github.com/shi3z/chainer_imagenet_tools
  - licensed under MIT
