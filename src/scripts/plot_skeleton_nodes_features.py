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

def draw_nodes(img, degrees, cmap, norm):
    img_copy = img.copy()
    nodes_coords = np.where(degrees > 2)

    for x, y in zip(*nodes_coords):
        color = (255 * np.array(cmap(norm(degrees[x,y]))))[:-1]
        img_copy = cv2.circle(img_copy, (y, x), radius=10, color=color, thickness=-1)
    
    return img_copy

def get_skeleton_nodes_features_plot(df_all, feature_name):
    df_sorted = df_all.sort_values(by=feature_name)
    fig, axes = plt.subplots(49, 5*2, figsize=(5*2*5, 49*5))
    max_degree = df_all['nodes_max'].max()

    cmap = cm.YlOrRd
    norm = Normalize(vmin=2, vmax=max_degree)


    for index, (img_name, row) in enumerate(df_sorted.iterrows()):
        print(img_name)
        sk_path = TEMPLATE_PATH.format(folder=row['folder'], 
                                       img_type="Skeletonized", 
                                       name=img_name, 
                                       suffix="j-SK")
        original_path = TEMPLATE_PATH.format(folder=row['folder'], 
                                            img_type="Original", 
                                            name=img_name, 
                                            suffix="j")
    
        sk_img = cv2.imread(sk_path)
        original_img = cv2.imread(original_path)
        
        if sk_img.shape > original_img.shape:
            sk_img = centutils.clip_image(sk_img, original_img.shape[:2])

        degrees, _ = ft.get_skeleton_data(sk_img)
        raw_data = degrees[np.where(degrees > 2)]

        data_img = draw_nodes(sk_img, degrees, cmap, norm)
        data_img = cv2.resize(data_img, IMG_SIZE)
        prev_mean = row['nodes_mean']
        prev_std = row['nodes_std']
        prev_max = row['nodes_max']
        prev_n = row['nodes_n']
        new_mean = raw_data.mean()
        new_std = raw_data.std()
        new_max = raw_data.max()
        new_n = len(raw_data)

        if not np.isnan(new_mean):
            assert abs(prev_mean - new_mean) < 1e-4
        if not np.isnan(new_std):
            assert abs(prev_std - new_std) < 1e-4
        if not np.isnan(new_max):
            assert abs(prev_max - new_max) < 1e-4
        assert prev_n == new_n 

        label = '{img_name}: n = {n}, max = {max}, mean = {mean:.2f}, std = {std:.2f}'.format(
            img_name=img_name, n=prev_n, max=prev_max, mean=prev_mean, std=prev_std)
        
        iii, jjj = (2*index) // 10, (2*index) % 10
        axes[iii, jjj].imshow(data_img)
        axes[iii, jjj].set_yticks([])
        axes[iii, jjj].set_xticks([])
        axes[iii, jjj].set_xlabel(label, fontsize=16, horizontalalignment='left', x=0.0, labelpad=20)

        bins = np.arange(3, max_degree + 1)
        pd.Series(raw_data).hist(bins=bins - 0.5, ax=axes[iii, jjj+1])
        axes[iii, jjj + 1].set_xticks(np.arange(3, 7))
        axes[iii, jjj + 1].grid(None)

        title = 'Skeletonized image highlighting the nodes, ordered by the feature {}'.format(
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
    
    feature_name = 'nodes_n'

    fig, axes = get_skeleton_nodes_features_plot( df_all, feature_name)
    fig.savefig(os.path.join(RESULTS_PATH, '{}.png'.format(feature_name)),
                    bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()