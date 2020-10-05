import os
import sys
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage.filters import uniform_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters.rank import entropy
from skimage.morphology import medial_axis
from skan import skeleton_to_csgraph, _testdata, draw, Skeleton, summarize
from . import centroids as centutils
from . import utils

def mask_results(results, jbin_img):
    results_copy = results.copy()
    results_copy[jbin_img == 0] = 0
    return results_copy

def window_stdev(img, window_size):
    X = img.astype(float)
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    variance = c2 - c1*c1
    variance = np.where(variance < 0, 0, variance)
    return np.sqrt(variance)

def voronoi_approx_statistics(img, raw_values=False):
    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(-distance, indices=False, footprint=np.ones((3, 3)), labels=imgx)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(distance, markers, watershed_line=True)
    mask = labels == 0
    distance_skel = distance.copy()
    distance_skel[~mask] = 0
    non_zero = distance_skel[mask]
    if raw_values:
        return non_zero.mean(), non_zero.std(), distance_skel
    return non_zero.mean(), non_zero.std()


def medial_axis_stadistics(img, raw_values=False):
    skel, distance = medial_axis(img, return_distance=True)
    dist_on_skel = distance * skel
    non_zero = dist_on_skel[np.where(dist_on_skel != 0)]
    if raw_values:
        return non_zero.mean(), non_zero.std(), dist_on_skel
    return non_zero.mean(), non_zero.std()

def std_filter_statistics(img, jbin_img, raw_values=False):
    filtered = window_stdev(img, window_size=7)
    mask_filtered = mask_results(filtered, jbin_img)
    non_zero = mask_filtered[np.where(mask_filtered != 0)]
    if raw_values:
        return non_zero.mean(), non_zero.std(), mask_filtered
    return non_zero.mean(), non_zero.std()

def entropy_filter_statistics(img, jbin_img, raw_values=False):
    filtered = entropy(img, np.ones((7,7)))
    mask_filtered = mask_results(filtered, jbin_img)
    non_zero = mask_filtered[np.where(mask_filtered != 0)]
    if raw_values:
        return non_zero.mean(), non_zero.std(), mask_filtered
    return non_zero.mean(), non_zero.std()

def get_skeleton_data(sk_img):
    pixel_graph, coordinates, degrees = skeleton_to_csgraph(sk_img)
    branch_data = summarize(Skeleton(sk_img))
    return degrees, branch_data


def get_area_features(nuclei_img, jbin_img):
    centroids = centutils.get_nuclei_centroids(nuclei_img)
    centroids = centutils.clean_centroids(centroids, jbin_img)
    centroids, moments = centutils.get_moments_cells(centroids, jbin_img)

    ret = {}

    cell_area = 0
    total_area = jbin_img.shape[0] * jbin_img.shape[1]

    for m in moments:
        cell_area += m['m00']

    white_area = len(jbin_img[jbin_img == 255])
    ret['cell_area_ratio'] = cell_area / total_area
    ret['white_area_ratio'] = white_area / total_area

    return ret


def get_skeleton_features(sk_img, raw_data=''):
    degrees, branch_data = get_skeleton_data(sk_img)

    ret = {}

    # End-to-end
    e2e_data = branch_data[branch_data['branch-type'] == 0]
    ret['e2e_n'] = len(e2e_data)
    ret['e2e_distance_mean'] = e2e_data['branch-distance'].mean()
    ret['e2e_distance_std'] = e2e_data['branch-distance'].std()
    ret['e2e_eu_distance_mean'] = e2e_data['euclidean-distance'].mean()
    ret['e2e_eu_distance_std'] = e2e_data['euclidean-distance'].std()
    ret['e2e_distance_ratio_mean'] = (e2e_data['euclidean-distance'] / e2e_data['branch-distance']).mean()
    ret['e2e_distance_ratio_std'] = (e2e_data['euclidean-distance'] / e2e_data['branch-distance']).std()

    # Junction-to-end
    j2e_data = branch_data[branch_data['branch-type'] == 1]
    ret['j2e_n'] = len(j2e_data)
    ret['j2e_distance_mean'] = j2e_data['branch-distance'].mean()
    ret['j2e_distance_std'] = j2e_data['branch-distance'].std()
    ret['j2e_eu_distance_mean'] = j2e_data['euclidean-distance'].mean()
    ret['j2e_eu_distance_std'] = j2e_data['euclidean-distance'].std()
    ret['j2e_distance_ratio_mean'] = (j2e_data['euclidean-distance'] / j2e_data['branch-distance']).mean()
    ret['j2e_distance_ratio_std'] = (j2e_data['euclidean-distance'] / j2e_data['branch-distance']).std()

    # Junction-to-junction
    j2j_data = branch_data[branch_data['branch-type'] == 2]
    ret['j2j_n'] = len(j2j_data)
    ret['j2j_distance_mean'] = j2j_data['branch-distance'].mean()
    ret['j2j_distance_std'] = j2j_data['branch-distance'].std()
    ret['j2j_eu_distance_mean'] = j2j_data['euclidean-distance'].mean()
    ret['j2j_eu_distance_std'] = j2j_data['euclidean-distance'].std()
    ret['j2j_distance_ratio_mean'] = (j2j_data['euclidean-distance'] / j2j_data['branch-distance']).mean()
    ret['j2j_distance_ratio_std'] = (j2j_data['euclidean-distance'] / j2j_data['branch-distance']).std()

    # Nodes
    nodes = degrees[np.where(degrees > 2)]
    ret['nodes_n'] = len(nodes)
    ret['nodes_max'] = nodes.max()
    ret['nodes_mean'] = nodes.mean()
    ret['nodes_std'] = nodes.std()

    return ret

def get_branch_thickness_features(jbin_img):
    ret = {}
    ma_mean, ma_std = medial_axis_stadistics(jbin_img)
    va_mean, va_std = voronoi_approx_statistics(jbin_img)

    ret['medial_axis_mean'] = ma_mean
    ret['medial_axis_std'] = ma_std
    ret['voronoi_approx_mean'] = va_mean
    ret['voronoi_approx_std'] = va_std

    return ret

def get_texture_features(original_img, jbin_img):
    ret = {}
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    jbin_img = cv2.cvtColor(jbin_img, cv2.COLOR_BGR2GRAY)
    sf_mean, sf_std = std_filter_statistics(original_img, jbin_img)
    ef_mean, ef_std = entropy_filter_statistics(original_img, jbin_img)

    ret['std_filter_mean'] = sf_mean
    ret['std_filter_std'] = sf_std
    ret['entropy_filter_mean'] = ef_mean
    ret['entropy_filter_std'] = ef_std

    return ret

def get_features(original_img, nuclei_img, jbin_img, jsk_img):
    """[summary]

    Args:
        original_img (np.ndarray, shape=(n,n), dtype=np.uint16): Cell-cell
            junction image, one channel.
        nuclei_img (np.ndarray, shape=(n,n), dtype=np.uint16): Nuclei image, one
            channel.
        jbin_img (np.ndarray, shape=(n,n), dtype=np.uint8): Segmented cell-cell
            junction image, binary image.
        jsk_img (np.ndarray, shape=(n,n), dtype=np.uint8): Skeletonized
            cell-cell junction image, binary image.

    Returns:
        dict[str->float]: computed features over the image.
    """
    features = {}

    # To compute the area functions, we use a opencv functions, and most of
    # those require 8-bit, 3 channels images.
    # nuclei_img_bgr = utils.convert_16_gray_to_8_bgr(nuclei_img)
    # jbin_img_bgr = cv2.cvtColor(jbin_img, cv2.COLOR_GRAY2BGR)

    #area_ft = get_area_features(nuclei_img_bgr, jbin_img)

    #sk_ft = get_skeleton_features(jsk_img)
    bt_ft = get_branch_thickness_features(jbin_img)
    # tx_ft = get_texture_features(original_img, jbin_img)

    #features.update(area_ft)
    #features.update(sk_ft)
    features.update(bt_ft)
    # features.update(tx_ft)

    return features