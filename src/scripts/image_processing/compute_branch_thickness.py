"""Script to process the images and compute the thickness of the branches"""

import os
import argparse
import tifffile
from ...lib import utils
from ...lib import image_processing as iputils


def main(imgs_dir, results_base_dir, algorithm):
    results_dir = os.path.join(results_base_dir,
                               'images',
                               f'branch_thickness_{algorithm}')
    os.makedirs(results_dir, exist_ok=True)
    functions_map = {
        "voronoi": iputils.branch_thickness_voronoi,
        "medial_axis": iputils.branch_thickness_medial_axis
    }

    for fullname in os.listdir(os.path.join(imgs_dir, 'CCJ')):
        if os.path.splitext(fullname)[1] != '.tif':
            continue
        img_name = fullname[:-5]
        _, _, seg_img, _ = utils.load_images(imgs_dir, img_name)
        processed = functions_map[algorithm](seg_img)
        processed_path = os.path.join(results_dir, f'{img_name}.tif')
        processed_32 = processed.astype('float32')
        tifffile.imsave(processed_path, processed_32)
        print(f'Processed image {img_name}')


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--imgs_dir',
        type=str,
        help="""Folder with the images. This folder must have 4 subfolders:
        CCJ, Nuclei, Segmented-CCJ and Skeletonized-CCJ.
        """,
        required=True
    )
    PARSER.add_argument(
        '--results_base_dir',
        type=str,
        help="""Base dir for the results. The results will actually be stored in
        the subfolder results_base_dir/images/branch_thickness_{algorithm}.""",
        required=True
    )
    PARSER.add_argument(
        '--algorithm',
        type=str,
        help="Algorithm to compute the thickness of the branches",
        choices=['voronoi', 'medial_axis'],
        required=True
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.imgs_dir, FLAGS.results_base_dir, FLAGS.algorithm)
