import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from src.lib import features as ft
from src.lib import centroids as centutils

TEMPLATE_PATH = 'data/{folder}/{img_type}/{name}{suffix}.tif'
RESULTS_PATH = 'results/features'
TYPE_SUFFIX_MAP = {"Original": "j", "Nuclei": "n", 
                   "Skeletonized": "j-SK", "Segmented": "j-BI"}
IMG_SIZE = (256, 256)


def get_texture_and_thickness_features_plot(df_all, feature_name):
    df_sorted = df_all.sort_values(by=feature_name)
    fig, axes = plt.subplots(49, 5*3, figsize=(5*3*5, 49*5))

    feature_name_no_stts = "_".join(feature_name.split("_")[:-1])
    for index, (img_name, row) in enumerate(df_sorted.iterrows()):
        print(img_name)
        seg_path = TEMPLATE_PATH.format(folder=row['folder'], 
                                       img_type="Segmented", 
                                       name=img_name, 
                                       suffix="j-BI")
        original_path = TEMPLATE_PATH.format(folder=row['folder'], 
                                            img_type="Original", 
                                            name=img_name, 
                                            suffix="j")
    
        seg_img = cv2.imread(seg_path)
        original_img = cv2.imread(original_path)
        
        if seg_img.shape > original_img.shape:
            seg_img = centutils.clip_image(seg_img, original_img.shape[:2])


        if feature_name_no_stts == 'medial_axis':
            title = 'Skeletonized image with the thickness of each point using the algorithm Medial Axis, ordered by mean of the distances'
            img_to_show = seg_img
            new_mean, new_std, data_img = ft.medial_axis_stadistics(
                seg_img, raw_values=True)
        elif feature_name_no_stts == 'voronoi_approx':   
            title = 'Skeletonized image with the thickness of each point using the algorithm Generalized Voronoi, ordered by mean of the distances'
            img_to_show = seg_img
            new_mean, new_std, data_img = ft.voronoi_approx_statistics(
                seg_img, raw_values=True)
        elif feature_name_no_stts == 'std_filter':    
            title = 'Texture computed over the segmented image using a std filter, ordered by the mean of the values'
            original_img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)    
            img_to_show = original_img
            new_mean, new_std, data_img = ft.std_filter_statistics(
                original_img_gray, seg_img_gray, raw_values=True)
        elif feature_name_no_stts == 'entropy_filter':  
            title = 'Texture computed over the segmented image using an entropy filter, ordered by the mean of the values'
            original_img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)    
            img_to_show = original_img
            new_mean, new_std, data_img = ft.entropy_filter_statistics(
                original_img_gray, seg_img_gray, raw_values=True)
        
        non_zero = data_img[np.where(data_img != 0)]

        prev_mean = row[feature_name_no_stts + '_mean']
        prev_std = row[feature_name_no_stts + '_std']
        
        assert abs(prev_mean - new_mean) < 1e-3
        assert abs(prev_std - new_std) < 1e-3
        assert abs(prev_mean - non_zero.mean()) < 1e-3
        assert abs(prev_std - non_zero.std()) < 1e-3

        data_img = cv2.resize(data_img, IMG_SIZE)
        img_to_show = cv2.resize(img_to_show, IMG_SIZE)

        label = '{img_name}: mean = {mean:.2f}, std = {std:.2f}'.format(
            img_name=img_name, mean=prev_mean, std=prev_std)
        
        iii, jjj = (3*index) // 15, (3*index) % 15

        axes[iii, jjj].imshow(img_to_show)
        axes[iii, jjj].set_yticks([])
        axes[iii, jjj].set_xticks([])

        axes[iii, jjj + 1].imshow(data_img)
        axes[iii, jjj + 1].set_yticks([])
        axes[iii, jjj + 1].set_xticks([])

        axes[iii, jjj].set_xlabel(label, fontsize=16, horizontalalignment='left', x=0.0, labelpad=20)

        pd.Series(non_zero).hist(bins=20, ax=axes[iii, jjj + 2])

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
    
    features_names = ['medial_axis_mean',
                      'voronoi_approx_mean',
                      'std_filter_mean',
                      'entropy_filter_mean']

    for feature_name in features_names:
        print("########################", feature_name, "########################")

        fig, axes = get_texture_and_thickness_features_plot(df_all, feature_name)
        fig.savefig(os.path.join(RESULTS_PATH, '{}.png'.format(feature_name)),
                        bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()

