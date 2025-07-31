"""Script to process the images and compute the texture of the CCJ"""

import os
import argparse
import tifffile
from ...lib import utils
from ...lib import image_processing as iputils


def main(imgs_dir, results_base_dir, algorithm, keep_previous):
    results_dir = os.path.join(
        results_base_dir, "images", f"texture_{algorithm}_filter"
    )
    os.makedirs(results_dir, exist_ok=True)
    functions_map = {
        "std": iputils.texture_std_filter,
        "entropy": iputils.texture_entropy_filter,
    }
    for fullname in os.listdir(os.path.join(imgs_dir, "Segmented-CCJ")):
        if os.path.splitext(fullname)[1] != ".tif":
            continue

        img_name = fullname[:-8]
        processed_path = os.path.join(results_dir, f"{img_name}.tif")

        if keep_previous and os.path.exists(processed_path):
            print(f"Already processed {img_name}")
            continue

        _, ccj_img, seg_img, _ = utils.load_images(imgs_dir, img_name)
        processed = functions_map[algorithm](ccj_img, seg_img)
        processed_32 = processed.astype("float32")
        tifffile.imsave(processed_path, processed_32)
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
        "--results_base_dir",
        type=str,
        help="""Base dir for the results. The results will actually be stored in
        the subfolder results_base_dir/images/texture_{algorithm}_filter.""",
        required=True,
    )
    PARSER.add_argument(
        "--algorithm",
        type=str,
        help="Algorithm to compute the texture of the CCJ",
        choices=["std", "entropy"],
        required=True,
    )
    PARSER.add_argument(
        "--keep_previous",
        type=bool,
        help="""True to keep previous results, False to overwrite""",
        default=True,
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.imgs_dir, FLAGS.results_base_dir, FLAGS.algorithm, FLAGS.keep_previous)
