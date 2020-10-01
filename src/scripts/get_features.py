import os
import cv2
import pandas as pd
from ..lib import features as ftutils
from ..lib import centroids as centutils



def main():
    ft_dict = {}
    save_path = 'results/NegsiRNA_features.csv'
    src_dir = 'data/NegsiRNA/'
    for fullname in os.listdir(os.path.join(src_dir, 'Original')):
        print("###################################################")
        if os.path.splitext(fullname)[1] != '.tif':
            continue
        name = fullname[:-5]
        print(name)
        if name != 'NegsiRNA_A1_NS':
            continue
        
        import pdb; pdb.set_trace()
        nuclei_path = os.path.join(src_dir,'Nuclei/{name}n.tif'.format(name=name))
        jbin_path = os.path.join(src_dir,'Segmented/{name}j-BI.tif'.format(name=name))
        jsk_path = os.path.join(src_dir,'Skeletonized/{name}j-SK.tif'.format(name=name))
        j_path = os.path.join(src_dir,'Original/{name}j.tif'.format(name=name))

        nuclei_img = cv2.imread(nuclei_path)
        jbin_img = cv2.imread(jbin_path)
        jsk_img = cv2.imread(jsk_path)
        original_img = cv2.imread(j_path)


        assert nuclei_img.shape == original_img.shape

        if jbin_img.shape > original_img.shape:
            jbin_img = centutils.clip_image(jbin_img, original_img.shape[:2])

        if jsk_img.shape > original_img.shape:
            jsk_img = centutils.clip_image(jsk_img, original_img.shape[:2])

        features = ftutils.get_features(original_img, nuclei_img, jbin_img, jsk_img)
        ft_dict[name] = features
    
    df = pd.DataFrame.from_dict(ft_dict, orient='index')
    df.to_csv(save_path)
    

if __name__ == '__main__':
    main()