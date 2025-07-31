import os
import cv2
import numpy as np


def crop_image(img, new_size):
    height, width = img.shape[:2]
    new_height, new_width = new_size
    i_coord = (height - new_height) // 2
    j_coord = (width - new_width) // 2
    return img[i_coord : i_coord + new_height, j_coord : j_coord + new_width]


def convert_16_gray_to_8_bgr(img16):
    img8 = (img16 / 256).astype("uint8")
    img8_bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    return img8_bgr


def convert_16_to_8(img16):
    img8 = (img16 / 256).astype("uint8")
    return img8


def load_images(imgs_dir, img_name):
    """Load all the versions of the same image.

    Args:
        imgs_dir (str): Base folder of the images.
        img_name (str): Name of the image.

    Returns:
        (np.ndarray(n,n), np.ndarray(n,n), np.ndarray(n,n), np.ndarray(n,n)):
            nuclei_img, ccj_img, seg_img, sk_img
    """
    nuclei_path = os.path.join(imgs_dir, f"Nuclei/{img_name}n.tif")
    ccj_path = os.path.join(imgs_dir, f"CCJ/{img_name}j.tif")
    seg_path = os.path.join(imgs_dir, f"Segmented-CCJ/{img_name}j-BI.tif")
    sk_path = os.path.join(imgs_dir, f"Skeletonized-CCJ/{img_name}j-SK.tif")

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


def resize_up_to(img, min_size):
    scale_factor = int(np.ceil(min_size / min(img.shape)))
    new_size = scale_factor * np.array(img.shape)
    return cv2.resize(img, (new_size[1], new_size[0]))


def contour_to_image(cnt):
    """Generates a binary image from a contour.

    Args:
        cnt (np.ndarray): Contour by its vertices in a np.ndarray with
            shape=(n_vertices, 2).
    Returns:
        np.ndarray, ndim=2, dtype=uint8: binary image with the filled
            contour.
    """
    bbox = cv2.boundingRect(cnt)
    norm_cnt = cnt - bbox[:2]
    img = np.zeros((bbox[3], bbox[2]), dtype="uint8")
    cv2.fillPoly(img, pts=[norm_cnt], color=255)
    return img


def draw_centroids(img, centroids):
    centroids_img = img.copy()
    for centroid in centroids:
        cv2.circle(centroids_img, tuple(centroid), 5, (0, 255, 0), -1)
    return centroids_img