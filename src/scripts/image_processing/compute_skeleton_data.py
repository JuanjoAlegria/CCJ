"""Script to process the images and compute the texture of the CCJ"""

import os
import argparse
import tifffile
from ...lib import utils
from ...lib import image_processing as iputils


def main(imgs_dir, results_base_dir, keep_previous):
    degrees_dir = os.path.join(results_base_dir, "images", "skeleton_degrees")
    dfs_dir = os.path.join(results_base_dir, "dataframes", "skeleton_data")
    os.makedirs(degrees_dir, exist_ok=True)
    os.makedirs(dfs_dir, exist_ok=True)

    for fullname in os.listdir(os.path.join(imgs_dir, "Segmented-CCJ")):
        if os.path.splitext(fullname)[1] != ".tif":
            continue
        img_name = fullname[:-8]
        degrees_path = os.path.join(degrees_dir, f"{img_name}.tif")
        df_path = os.path.join(dfs_dir, f"{img_name}.csv")

        if keep_previous and os.path.exists(df_path):
            print(f"Already processed {img_name}")
            continue

        _, _, _, sk_img = utils.load_images(imgs_dir, img_name)
        _, _, degrees, branch_data = iputils.skeleton_data(sk_img)

        degrees_8 = degrees.astype("uint8")
        tifffile.imsave(degrees_path, degrees_8)
        branch_data.to_csv(df_path)
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
        two subfolders:
            - results_base_dir/images/skeleton_degrees: images where each point
            of the skeleton has its degree, and
            - results_base_dir/dataframes/skeleton_data: dataframes with the
            summarized information of the skeleton
            """,
        required=True,
    )
    PARSER.add_argument(
        "--keep_previous",
        type=bool,
        help="""True to keep previous results, False to overwrite""",
        default=True,
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.imgs_dir, FLAGS.results_base_dir, FLAGS.keep_previous)
