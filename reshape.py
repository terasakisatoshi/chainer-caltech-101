import cv2
import argparse
from glob import glob
import os
import numpy


output_side_length = 256


def reshape(args):
    for source_imgpath in os.listdir(args.source_dir):
        pattern = os.path.join(args.source_dir, source_imgpath, "*.jpg")
        files = glob(pattern)
        for file in files:
            img = cv2.imread(file)
            height, width, depth = img.shape
            new_height = output_side_length
            new_width = output_side_length
            if height > width:
                new_height = int(output_side_length * height / width)
            else:
                new_width = int(output_side_length * width / height)
            resized_img = cv2.resize(img, (new_width, new_height))
            height_offset = (new_height - output_side_length) // 2
            width_offset = (new_width - output_side_length) // 2
            cropped_img = resized_img[height_offset:height_offset + output_side_length,
                                      width_offset:width_offset + output_side_length]
            assert cropped_img.shape == (256, 256, 3)
            save_path = os.path.join(
                args.target_dir, source_imgpath, os.path.basename(file))
            if not os.path.exists(os.path.join(args.target_dir, source_imgpath)):
                os.makedirs(os.path.join(args.target_dir, source_imgpath))
            cv2.imwrite(save_path, cropped_img)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="101_ObjectCategories")
    parser.add_argument("--target_dir", default="reshaped")
    args = parser.parse_args()
    return args


def main():
    args = parse_argument()
    reshape(args)


if __name__ == '__main__':
    main()
