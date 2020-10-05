import os
import cv2
import numpy as np
from skimage import filters
from scipy.ndimage.filters import uniform_filter

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

def window_stdev(img, window_size):
    """Computes the standard deviation in the neighborhood of each pixel.

    Args:
        img (np.ndarray, shape=(n,n), dtype=np.uint8): Image where we the filter
            is going to be applied.
        window_size (int): Size of filter (equal in x and y)

    Returns:
        np.ndarray, shape=(n,n): Image after the filter is applied.
    """
    X = img.astype(float)
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    variance = c2 - c1*c1
    variance = np.where(variance < 0, 0, variance)
    return np.sqrt(variance)


def load_images(imgs_dir, img_name):
    nuclei_path = os.path.join(imgs_dir, f'Nuclei/{img_name}n.tif')
    original_path = os.path.join(imgs_dir, f'Original/{img_name}j.tif')
    seg_path = os.path.join(imgs_dir, f'Segmented/{img_name}j-BI.tif')
    sk_path = os.path.join(imgs_dir, f'Skeletonized/{img_name}j-SK.tif')

    nuclei_img = cv2.imread(nuclei_path, cv2.IMREAD_UNCHANGED)
    original_img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    sk_img = cv2.imread(sk_path, cv2.IMREAD_UNCHANGED)


    assert nuclei_img.shape == original_img.shape

    if seg_img.shape > original_img.shape:
        seg_img = crop_image(seg_img, original_img.shape[:2])

    if sk_img.shape > original_img.shape:
        sk_img = crop_image(sk_img, original_img.shape[:2])

    return nuclei_img, original_img, seg_img, sk_img