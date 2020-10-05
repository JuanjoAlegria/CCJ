import os
import argparse
import pandas as pd
from ..lib import features as ftutils
from ..lib import utils


def main(imgs_dir, save_path):
    import pdb; pdb.set_trace()
    ft_dict = {}

    for fullname in os.listdir(os.path.join(imgs_dir, 'Original')):
        if os.path.splitext(fullname)[1] != '.tif':
            continue
        img_name = fullname[:-5]
        print("###################################################")
        print(img_name)

        nuclei_img, original_img, seg_img, sk_img = utils.load_images(imgs_dir,
                                                                      img_name)

        features = ftutils.get_features(original_img, nuclei_img,
                                        seg_img, sk_img)
        ft_dict[img_name] = features

    df = pd.DataFrame.from_dict(ft_dict, orient='index')
    df.to_csv(save_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--imgs_dir',
        type=str,
        help="""Folder with the images. This folder must have 4 subfolders:
        Original, Nuclei, Segmented and Skeletonized.
        """,
        required=True
    )
    PARSER.add_argument(
        '--save_path',
        type=str,
        help="Path where the resultin CSV will be stored",
        required=True
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.imgs_dir, FLAGS.save_path)
