import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..lib import features as ftutils 

TEMPLATE_PATH = "data/{folder}/{img_type}/{name}{suffix}.tif"
RESULTS_PATH = 'results/features'
TYPE_SUFFIX_MAP = {"Original": "j", "Nuclei": "n", 
                   "Skeletonized": "j-SK", "Segmented": "j-BI"}
IMG_SIZE = (256, 256)
SUBIMG_PADDING = 10 


def get_subimage(images):
    if len(images) == 2:
        subimg = 255*np.ones((IMG_SIZE[0], IMG_SIZE[1]*2 + SUBIMG_PADDING, 3), dtype=seg_img.dtype)
        subimg[:, :IMG_SIZE[1]] = images[0]
        subimg[:, IMG_SIZE[1]+SUBIMG_PADDING:] = images[1]
        return subimg
          


def draw_lines(df, image, color, thickness=5):
    copy = image.copy()
    for _, row in df.iterrows():
        x0 = int(row['image-coord-src-0'])
        y0 = int(row['image-coord-src-1'])
        x1 = int(row['image-coord-dst-0'])
        y1 = int(row['image-coord-dst-1'])
        cv2.line(copy, (y0, x0), (y1, x1), color, thickness)
    return copy


def get_single_value_features_plot(df_all, feature_name, required_images): 
    df_sorted = df_all.sort_values(by=feature_name)
    fig, axes = plt.subplots(49, 5, figsize=(5*2*5, 49*5))
    for index, (img_name, row) in enumerate(df_sorted.iterrows()):
        imgs_paths = [TEMPLATE_PATH.format(folder=row['folder'], 
                                           img_type=img_type, 
                                           name=img_name, 
                                           suffix=TYPE_SUFFIX_MAP[img_type])
                     for img_type in required_images]
        imgs = [cv2.imread(img_path) for imgs_path in imgs_paths]
        imgs = [cv2.resize(img, IMG_SIZE) for img in imgs]
        
        img = 255*np.ones((256, 256*2 + 10, 3), dtype=seg_img.dtype)
        img[:, :256] = seg_img
        img[:, 256+10:] = nuclei_img
        
        iii, jjj = index // 5, index % 5
        axes[iii, jjj].imshow(img)
        axes[iii, jjj].set_yticks([])
        axes[iii, jjj].set_xticks([])
        axes[iii, jjj].set_xlabel('{img_name}: {war:.3f}'.format(img_name=img_name, war=row[feature_name]), fontsize=20)
        
    suptitle = fig.suptitle('Segmented and nuclei images, ordered by the feature{}'.format(feature_name), y=1.005, fontsize=30)
    fig.tight_layout()
    fig.savefig('../results/cell_area_ratio.png', bbox_extra_artists=(suptitle,), bbox_inches="tight")


def get_skeleton_branches_features_plot(df_all, feature_name):
    branch_to_number_map = {"e2e": 0, "j2e": 1, "j2j": 2}
    branch_to_colors_map = {
        "e2e": (255, 0, 0),
        "j2e": (0, 255, 0),
        "j2j": (0, 0, 255)    
    }
    branch_to_thickness_map = {"e2e": 30, "j2e": 10, "j2j": 10}
    ft_to_column = {
        "distance": "branch-distance",
        "eu_distance": "euclidean-distance",
        "distance_ratio": "distance-ratio"
    }

    ft_prefix = feature_name[:3]
    ft_number = branch_to_number_map[ft_prefix]
    ft_color = branch_to_colors_map[ft_prefix]
    ft_thickness = branch_to_thickness_map[ft_prefix]
    ft_type = ft_to_column['_'.join(feature_name.split("_")[1:-1])]

    df_sorted = df_all.sort_values(by=feature_name)
    fig, axes = plt.subplots(49, 10, figsize=(10*5, 49*5))
    
    for index, (img_name, row) in enumerate(df_sorted.iterrows()):
        seg_img = TEMPLATE_PATH.format(folder=row['folder'], 
                                       img_type="Segmented", 
                                       name=img_name, 
                                       suffix="j-BI")
        sk_img = TEMPLATE_PATH.format(folder=row['folder'], 
                                       img_type="Skeletonized", 
                                       name=img_name, 
                                       suffix="j-SK")
        degrees, branch_data = ftutils.get_skeleton_data(sk_img)
        branch_data['distance-ratio'] = branch_data['euclidean-distance'] / branch_data['branch-distance']
        branch_data_slice = branch_data['branch-type' == ft_number]
        data_img = draw_lines(df=branch_data_slice,
                                image=seg_img,
                                color=ft_color,
                                thickness=ft_thickness)
        data_img = cv2.imresize(data_img, IMG_SIZE)
        raw_data = branch_data_slice[ft_type]
        feature_name_no_stts = "_".join(feature_name.split("_")[:-1])
        prev_mean = row[feature_name_no_stts + '_mean']
        prev_mean = row[feature_name_no_stts + 'std']
        new_mean = raw_data.mean()
        new_std = raw_data.std()

        assert abs(prev_mean - new_mean) < 1e-4
        assert abs(prev_std - new_std) < 1e-4
        
        label = '{img_name}: mean = {mean:.3f}, std = {std:.3f}'.format(
            img_name=img_name, mean=prev_mean, std=prev_std)
        
        iii, jjj = (2*index) // 10, (2*index) % 10
        axes[iii, jjj].imshow(data_img)
        raw_data.hist(bins=20, ax=axes[iii, jjj+1])


        axes[iii, jjj].set_yticks([])
        axes[iii, jjj].set_xticks([])
        axes[iii, jjj+1].set_yticks([])
        axes[iii, jjj+1].set_yticks([])

        axes[iii, jjj].set_xlabel(label, fontsize=20)
        
    suptitle = fig.suptitle('Segmented image with overlay skeleton, ordered by the feature {}'.format(feature_name), y=1.005, fontsize=30)
    fig.tight_layout()
    fig.savefig(os.join.path(RESULTS_PATH, '{}.png'.format(feature_name)), 
                bbox_extra_artists=(suptitle,), bbox_inches="tight")

