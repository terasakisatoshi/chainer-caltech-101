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

# Appendix

I uploaded output files `train.txt, test.txt, label.txt mean.npy` and pretraind model `pretraind_googlenet.npz` in `./example`

```
python predict.py --trained ./example
```

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
