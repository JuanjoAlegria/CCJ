import os
import cv2

def crop_image(img, new_size):
    height, width = img.shape[:2]
    new_height, new_width = new_size
    i_coord = (height - new_height) // 2
    j_coord = (width - new_width) // 2
    return img[i_coord: i_coord + new_height, j_coord: j_coord + new_width]

def convert_16_gray_to_8_bgr(img16):
    img8 = (img16 / 256).astype('uint8')
    img8_bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    return img8_bgr

def convert_16_to_8(img16):
    img8 = (img16 / 256).astype('uint8')
    return img8


def load_images(imgs_dir, img_name):
    nuclei_path = os.path.join(imgs_dir, f'Nuclei/{img_name}n.tif')
    ccj_path = os.path.join(imgs_dir, f'CCJ/{img_name}j.tif')
    seg_path = os.path.join(imgs_dir, f'Segmented-CCJ/{img_name}j-BI.tif')
    sk_path = os.path.join(imgs_dir, f'Skeletonized-CCJ/{img_name}j-SK.tif')

    nuclei_img = cv2.imread(nuclei_path, cv2.IMREAD_UNCHANGED)
    ccj_img = cv2.imread(ccj_path, cv2.IMREAD_UNCHANGED)
    seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    sk_img = cv2.imread(sk_path, cv2.IMREAD_UNCHANGED)


    assert nuclei_img.shape == ccj_img.shape

    if seg_img.shape > ccj_img.shape:
        seg_img = crop_image(seg_img, ccj_img.shape[:2])

    if sk_img.shape > ccj_img.shape:
        sk_img = crop_image(sk_img, ccj_img.shape[:2])

    return nuclei_img, ccj_img, seg_img, sk_img