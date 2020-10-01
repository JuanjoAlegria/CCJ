import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.lib import features as ft
from src.lib import centroids as centutils

TEMPLATE_PATH = 'data/{folder}/{img_type}/{name}{suffix}.tif'
RESULTS_PATH = 'results/features'
TYPE_SUFFIX_MAP = {"Original": "j", "Nuclei": "n", 
                   "Skeletonized": "j-SK", "Segmented": "j-BI"}
IMG_SIZE = (256, 256)
SUBIMG_PADDING = 10 
BRANCH_TO_NUMBER_MAP = {"e2e": 0, "j2e": 1, "j2j": 2}
BRANCH_TO_COLORS_MAP = {
    "e2e": (255, 0, 0),
    "j2e": (0, 255, 0),
    "j2j": (0, 0, 255)    
}
BRANCH_TO_THICKNESS_MAP = {"e2e": 30, "j2e": 10, "j2j": 10}
FT_TO_COLUMN = {
    "distance": "branch-distance",
    "eu_distance": "euclidean-distance",
    "distance_ratio": "distance-ratio"
}


def draw_lines(df, image, color, thickness=5):
    copy = image.copy()
    for _, row in df.iterrows():
        x0 = int(row['image-coord-src-0'])
        y0 = int(row['image-coord-src-1'])
        x1 = int(row['image-coord-dst-0'])
        y1 = int(row['image-coord-dst-1'])
        cv2.line(copy, (y0, x0), (y1, x1), color, thickness)
    return copy

def get_skeleton_branches_features_plot(df_all, feature_name):

    ft_prefix = feature_name[:3]
    ft_number = BRANCH_TO_NUMBER_MAP[ft_prefix]
    ft_color = BRANCH_TO_COLORS_MAP[ft_prefix]
    ft_thickness = BRANCH_TO_THICKNESS_MAP[ft_prefix]
    ft_type = FT_TO_COLUMN['_'.join(feature_name.split("_")[1:-1])]

    df_sorted = df_all.sort_values(by=feature_name)
    fig, axes = plt.subplots(49, 5*2, figsize=(5*2*5, 49*5))
    
    for index, (img_name, row) in enumerate(df_sorted.iterrows()):
        print(img_name)
        seg_path = TEMPLATE_PATH.format(folder=row['folder'], 
                                       img_type="Segmented", 
                                       name=img_name, 
                                       suffix="j-BI")
        sk_path = TEMPLATE_PATH.format(folder=row['folder'], 
                                       img_type="Skeletonized", 
                                       name=img_name, 
                                       suffix="j-SK")
        original_path = TEMPLATE_PATH.format(folder=row['folder'], 
                                            img_type="Original", 
                                            name=img_name, 
                                            suffix="j")
        
        seg_img = cv2.imread(seg_path)
        sk_img = cv2.imread(sk_path)
        original_img = cv2.imread(original_path)
        
        if seg_img.shape > original_img.shape:
            seg_img = centutils.clip_image(seg_img, original_img.shape[:2])

        if sk_img.shape > original_img.shape:
            sk_img = centutils.clip_image(sk_img, original_img.shape[:2])

        degrees, branch_data = ft.get_skeleton_data(sk_img)
        branch_data['distance-ratio'] = branch_data['euclidean-distance'] / branch_data['branch-distance']
        branch_data_slice = branch_data[branch_data['branch-type'] == ft_number]
        data_img = draw_lines(df=branch_data_slice,
                                image=seg_img,
                                color=ft_color,
                                thickness=ft_thickness)
        data_img = cv2.resize(data_img, IMG_SIZE)
        raw_data = branch_data_slice[ft_type]
        feature_name_no_stts = "_".join(feature_name.split("_")[:-1])
        prev_mean = row[feature_name_no_stts + '_mean']
        prev_std = row[feature_name_no_stts + '_std']
        new_mean = raw_data.mean()
        new_std = raw_data.std()

        if not np.isnan(new_mean):
            assert abs(prev_mean - new_mean) < 1e-4
        if not np.isnan(new_std):
            assert abs(prev_std - new_std) < 1e-4
        
        label = '{img_name}: n = {n}, mean = {mean:.2f}, std = {std:.2f}'.format(
            img_name=img_name, n=len(raw_data), mean=prev_mean, std=prev_std)
        
        iii, jjj = (2*index) // 10, (2*index) % 10
        axes[iii, jjj].imshow(data_img)
        raw_data.hist(bins=20, ax=axes[iii, jjj+1])


        axes[iii, jjj].set_yticks([])
        axes[iii, jjj].set_xticks([])

        axes[iii, jjj].set_xlabel(label, fontsize=16, horizontalalignment='left', x=0.0, labelpad=20)
        title = 'Segmented image with overlay skeleton, ordered by the feature {}'.format(
        feature_name)    
    fig.suptitle(title, fontsize=30, y=0.89)
    return fig, axes


def main():
    df_first_path = 'results/first_features.csv'
    df_negsi_path = 'results/NegsiRNA_features.csv'
    df_test_path = 'results/Test_features.csv'

    df_first = pd.read_csv(df_first_path, index_col=0)
    df_negsi = pd.read_csv(df_negsi_path, index_col=0)
    df_test = pd.read_csv(df_test_path, index_col=0)

    df_inter_first_negsi = df_first.merge(df_negsi, left_index=True, right_index=True)
    df_inter_first_test = df_first.merge(df_test, left_index=True, right_index=True)
    df_inter_negsi_test = df_negsi.merge(df_test, left_index=True, right_index=True)

    df_test_no_intersection = df_test.drop(df_inter_first_test.index)

    df_first['folder'] = 'first'
    df_negsi['folder'] = 'NegsiRNA'
    df_test_no_intersection['folder'] = 'Test'

    df_all = pd.concat([df_first, df_negsi, df_test_no_intersection])
    df_all.fillna(0)    
    skeleton_features = ['j2e_distance_mean',
                         'j2e_eu_distance_mean', 
                         'j2e_distance_ratio_mean', 
                         'j2j_distance_mean',
                         'j2j_eu_distance_mean',
                         'j2j_distance_ratio_mean',
                         'e2e_distance_mean', 
                         'e2e_eu_distance_mean',
                         'e2e_distance_ratio_mean']

    for feature_name in skeleton_features:
        print("########################", feature_name, "########################")
        fig, axes = get_skeleton_branches_features_plot(
                                    df_all, feature_name)
        fig.savefig(os.path.join(RESULTS_PATH, '{}.png'.format(feature_name)),
                    bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()