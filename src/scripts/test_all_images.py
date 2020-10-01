import os
import cv2
import matplotlib.pyplot as plt
from ..lib import centroids as centutils

# FILENAMES = ['ACTB_B2_CA', 'ACTB_B2_NO', 'ACTB_B2_NS', 'ACTN4_L2_NO', 
#          'ACTN4_L3_NO', 'ACTR3_C3_NO', 'ADAM9_C2_NO', 'ADD3_A2_CA',
#          'AKAP5_B2_CA', 'AKT1_B1_CA', 'AKT1_B2_CA', 'NegsiRNA_A2_NS',
#          'NegsiRNA_A2b_CA', 'NegsiRNA_A2b_NO', 'NegsiRNA_A2b_NS', 'NegsiRNA_A3_CA',
#          'NegsiRNA_A3_NO', 'NegsiRNA_A3_NS', 'NegsiRNA_B1_CA', 'NegsiRNA_B1_NO']


def main():
    src_dir = 'data/Test/'
    dst_dir = 'results/Test/'
    os.makedirs(dst_dir, exist_ok=True)
    for fullname in os.listdir(os.path.join(src_dir, 'Original')):
        print("###################################################")
        name = fullname[:-5]
        print(name)
        nuclei_path = os.path.join(src_dir,'Nuclei/{name}n.tif'.format(name=name))
        jbin_path = os.path.join(src_dir,'Segmented/{name}j-BI.tif'.format(name=name))
        jsk_path = os.path.join(src_dir,'Skeletonized/{name}j-SK.tif'.format(name=name))
        j_path = os.path.join(src_dir,'Original/{name}j.tif'.format(name=name))

        nuclei_img = cv2.imread(nuclei_path)
        jbin_img = cv2.imread(jbin_path)
        jsk_img = cv2.imread(jsk_path)
        j_img = cv2.imread(j_path)

        centroids = centutils.get_nuclei_centroids(nuclei_img)
        centroids = centutils.clean_centroids(centroids, jbin_img)
        centroids, moments = centutils.get_moments_cells(centroids, jbin_img)

        bin_centroid_final = centutils.draw_centroids(centroids, jbin_img)
        nuclei_centroid_final = centutils.draw_centroids(centroids, nuclei_img)

        fig, ax = plt.subplots(1, 2, figsize=(30, 15))
        ax[0].imshow(bin_centroid_final)
        ax[1].imshow(nuclei_centroid_final)

        for centroid, moment in zip(centroids, moments):
            area = moment['m00']
            ax[0].text(centroid[0], centroid[1], str(area), color='lightblue')
            ax[1].text(centroid[0], centroid[1], str(area), color='lightblue')

        plt.tight_layout()
        fig.savefig(os.path.join(dst_dir, '{name}_centroids_area.png'.format(name=name)))
        plt.close("all")


if __name__ == '__main__':
    main()