"""
create file interface that is available for chainer imagenet
usage: 
$ python create_dataset.py path/to/dataset
e.g. $ python create_dataset.py 101_ObjectCategories
"""
import argparse
from glob import glob
import os


def create_dataset(args):
    train_file = open("train.txt", 'w')
    test_file = open("test.txt", 'w')
    label_file = open("label.txt", 'w')
    dataset_path = os.path.abspath(args.dataset)
    for class_idx, label in enumerate(os.scandir(dataset_path)):
        label_file.write(label.name + "\n")

        image_paths = glob(os.path.join(label.path, "*.jpg"))
        length = len(image_paths)

        for path in image_paths[:int(length * 0.75)]:
            train_file.write(path + " %d\n" % class_idx)

        for path in image_paths[int(length * 0.75):]:
            test_file.write(path + " %d\n" % class_idx)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", help="path to dataset")
    return parser.parse_args()


def main():
    args = parse_argument()
    create_dataset(args)


if __name__ == '__main__':
    main()
