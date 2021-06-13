"""Script to process the images and compute the texture of the CCJ"""

import os
import argparse
import tifffile
from ...lib import utils
from ...lib import image_processing as iputils


def main(imgs_dir, results_dir, keep_previous):
    os.makedirs(results_dir, exist_ok=True)
    for fullname in os.listdir(os.path.join(imgs_dir, "Segmented-CCJ")):
        if os.path.splitext(fullname)[1] != ".tif":
            continue

        img_name = fullname[:-8]
        img_path = os.path.join(results_dir, f"{img_name}.tif")

        if keep_previous and os.path.exists(img_path):
            print(f"Already processed {img_name}")
            continue

        _, ccj_img, seg_img, _ = utils.load_images(imgs_dir, img_name)
        masked = iputils.get_mask(ccj_img, seg_img)
        tifffile.imsave(img_path, masked)
        print(f"Processed image {img_name}")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--imgs_dir",
        type=str,
        help="""Folder with the images. This folder must have 4 subfolders:
        CCJ, Nuclei, Segmented-CCJ and Skeletonized-CCJ.
        """,
        required=True,
    )
    PARSER.add_argument(
        "--results_dir",
        type=str,
        help="""Folder where the results will be stored""",
        required=True,
    )
    PARSER.add_argument(
        "--keep_previous",
        type=bool,
        help="""True to keep previous results, False to overwrite""",
        default=True,
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.imgs_dir, FLAGS.results_dir, FLAGS.keep_previous)
