# chainer-imagenet
imagenet implemented chainer including prepare dataset scripts

# usage
Do the following procedures (See also my documents on Qiita). 

- https://qiita.com/SatoshiTerasaki/items/c0a5a25b8bb82e95371b

## download original dataset

```console
$ wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
```

## reshape image data

```console
$ python reshape.py --source_dir 101_ObjectCategories --target_dir reshaped
```

## create dataset file
```console
$ python create_dataset.py reshaped
```
This creates `train.txt, test.txt, label.txt`

## calc mean

```console
$ python compute_mean.py train.txt
```

## train google net
```console
$ python train.py train.txt test.txt -a googlenet
```

# misc

I uploaded output files `train.txt, test.txt, label.txt mean.npy` and pretraind model `pretraind_googlenet.npz` in `/example`

# references

- https://github.com/chainer/chainer/tree/master/examples/imagenet
- https://github.com/shi3z/chainer_imagenet_tools
- http://www.image-net.org/