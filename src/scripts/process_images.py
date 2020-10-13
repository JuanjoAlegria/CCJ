"""Script to process the images and compute the thickness of the branches
and the texture of the images"""

import os
import argparse
import tifffile
from ..lib import utils
from ..lib import image_processing as iputils


def main(imgs_dir, results_dir, feature):
    feature_dir = os.path.join(results_dir, feature)
    os.makedirs(feature_dir, exist_ok=True)
    for fullname in os.listdir(os.path.join(imgs_dir, 'CCJ')):
        if os.path.splitext(fullname)[1] != '.tif':
            continue
        img_name = fullname[:-5]
        _, ccj_img, seg_img, _ = utils.load_images(imgs_dir, img_name)
        if feature == 'branch_thickness_voronoi':
            processed = iputils.branch_thickness_voronoi(seg_img)
        elif feature == 'branch_thickness_medial_axis':
            processed = iputils.branch_thickness_medial_axis(seg_img)
        elif feature == 'texture_entropy_filter':
            processed = iputils.texture_entropy_filter(ccj_img, seg_img)
        elif feature == 'texture_std_filter':
            processed = iputils.texture_std_filter(ccj_img, seg_img)

        processed_path = os.path.join(feature_dir, f'{img_name}.tif')
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
        '--results_dir',
        type=str,
        help="Folder where the results should be stored.",
        required=True
    )
    PARSER.add_argument(
        '--feature',
        type=str,
        help="Feature to compute",
        choices=['branch_thickness_voronoi',
                 'branch_thickness_medial_axis',
                 'texture_entropy_filter',
                 'texture_std_filter'],
        required=True
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.imgs_dir, FLAGS.results_dir, FLAGS.feature)
