import os
import argparse
import cv2
import pandas as pd
from ..lib import features as ftutils
from ..lib import utils


def main(raw_dir, interim_dir, save_path):
    ft_dict = {}

    raw_imgs_dir = os.path.join(raw_dir, 'images')

    for fullname in os.listdir(os.path.join(raw_imgs_dir, 'CCJ')):
        if os.path.splitext(fullname)[1] != '.tif':
            continue
        img_name = fullname[:-5]
        print("###################################################")
        print(img_name)

        # We load the segmented image in this way, because there are some
        # segmented images whose size differs from the ccj_img size,
        # so that condition must be checked and fixed before using the image.

        _, _, seg_img, _  = utils.load_images(raw_imgs_dir, img_name)

        template_img_path = os.path.join(interim_dir, 'images',
                                         '{ft_name}', f'{img_name}.tif')
        template_df_path = os.path.join(interim_dir, 'dataframes',
                                        '{ft_name}', f'{img_name}.csv')
        centroids_data = pd.read_csv(
            template_df_path.format(ft_name='centroids_data'), index_col=0)
        blobs_data = pd.read_csv(
            template_df_path.format(ft_name='blobs_data'), index_col=0)
        skeleton_data = pd.read_csv(
            template_df_path.format(ft_name='skeleton_data'), index_col=0)
        degrees_img = cv2.imread(
            template_img_path.format(ft_name='skeleton_degrees'),
            cv2.IMREAD_UNCHANGED)
        bt_medial_axis_img = cv2.imread(
            template_img_path.format(ft_name='branch_thickness_medial_axis'),
            cv2.IMREAD_UNCHANGED)
        bt_voronoi_img = cv2.imread(
            template_img_path.format(ft_name='branch_thickness_voronoi'),
            cv2.IMREAD_UNCHANGED)
        tx_std_img = cv2.imread(
            template_img_path.format(ft_name='texture_std_filter'),
            cv2.IMREAD_UNCHANGED)
        tx_entropy_img = cv2.imread(
            template_img_path.format(ft_name='texture_entropy_filter'),
            cv2.IMREAD_UNCHANGED)


        features = ftutils.get_features(seg_img, centroids_data, blobs_data,
                                        skeleton_data, degrees_img,
                                        bt_medial_axis_img, bt_voronoi_img,
                                        tx_std_img, tx_entropy_img)
        ft_dict[img_name] = features

    df = pd.DataFrame.from_dict(ft_dict, orient='index')
    df.to_csv(save_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--raw_dir',
        type=str,
        help="""Base folder with the raw data. It is expected that this folder
        have a subfolder named images, and inside that subfolder another 4
        subfolders: CCJ, Nuclei, Segmented-CCJ and Skeletonized-CCJ.
        """,
        required=True
    )
    PARSER.add_argument(
        '--interim_dir',
        type=str,
        help="""Base folder with the interim data, in subfolders images and
        dataframes""",
        required=True
    )
    PARSER.add_argument(
        '--save_path',
        type=str,
        help="Path where the resultin CSV will be stored",
        required=True
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.raw_dir, FLAGS.interim_dir, FLAGS.save_path)
