"""Script to process the images and compute the texture of the CCJ"""

import os
import argparse
from ...lib import utils
from ...lib import cell_blobs_skimage as blobsutils


def main(imgs_dir, results_base_dir, keep_previous):
    dfs_dir = os.path.join(results_base_dir, "dataframes", "blobs_data")
    os.makedirs(dfs_dir, exist_ok=True)

    for fullname in os.listdir(os.path.join(imgs_dir, "Segmented-CCJ")):
        if os.path.splitext(fullname)[1] != ".tif":
            continue
        img_name = fullname[:-8]
        df_path = os.path.join(dfs_dir, f"{img_name}.csv")

        if keep_previous and os.path.exists(df_path):
            print(f"Already processed {img_name}")
            continue

        _, _, seg_img, _ = utils.load_images(imgs_dir, img_name)
        blobs_df = blobsutils.get_blobs_measurements(seg_img)
        blobs_df.to_csv(df_path)
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
        the subfolder results_base_dir/dataframes/blobs_data.
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
