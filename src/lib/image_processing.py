"""Module with functions to process the images."""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage.filters import uniform_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters.rank import entropy
from skimage.morphology import medial_axis
from skan import skeleton_to_csgraph, summarize, Skeleton
from . import utils

def mask_results(results_img, seg_img):
    """Applies a mask over the results of a operation.

    Args:
        results_img (np.array, shape=(n,n)): Image with the results of an
            operation.
        seg_img (np.array, shape=(n,n)): Mask.

    Returns:
        np.array, shape=(n,n): Masked image.
    """
    results_copy = results_img.copy()
    results_copy[seg_img == 0] = 0
    return results_copy

def window_stdev(img, window_size):
    """Computes the standard deviation in the neighborhood of each pixel.

    For more information, go to
    https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html .

    Args:
        img (np.ndarray, shape=(n,n), dtype=np.uint8): Image where we the filter
            is going to be applied.
        window_size (int): Size of filter (equal in x and y)

    Returns:
        np.ndarray, shape=(n,n): Image after the filter is applied.
    """
    img_float = img.astype(float)
    mean_x = uniform_filter(img_float, window_size, mode='reflect')
    mean_xx = uniform_filter(img_float*img_float, window_size, mode='reflect')
    variance = mean_xx - mean_x*mean_x
    variance = np.where(variance < 0, 0, variance)
    return np.sqrt(variance)


def branch_thickness_voronoi(seg_img):
    """Computes the thickness of the branches using an approximation of the
    generalized Voronoi algorithm.

    Args:
        seg_img (np.array, shape=(n,n)): Binarized cell-cell junction image.

    Returns:
        np.array, shape=(n,n), dtype=float: Resulting image, where each point
            of the skeleton has the thickness in that point.
    """
    distance = ndi.distance_transform_edt(seg_img)
    local_maxi = peak_local_max(-distance, indices=False,
                                footprint=np.ones((3, 3)), labels=seg_img)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(distance, markers, watershed_line=True)
    mask = labels == 0
    distance_on_skel = distance.copy()
    distance_on_skel[~mask] = 0
    return distance_on_skel

def branch_thickness_medial_axis(seg_img):
    """Computes the thickness of the branches using the medial axis algorithm.

    Args:
        seg_img (np.array, shape=(n,n)): Binarized cell-cell junction image.

    Returns:
        np.array, shape=(n,n), dtype=float: Resulting image, where each point
            of the skeleton has the thickness in that point.
    """
    skel, distance = medial_axis(seg_img, return_distance=True)
    dist_on_skel = distance * skel
    return dist_on_skel

def texture_std_filter(ccj_img, seg_img, window_size=7):
    """Applies a texture filter over ccj_img, and then uses seg_img to mask those
    results. The texture is computed using a standard deviation filter.

    Args:
        ccj_img (np.array, shape=(n,n), dtype=np.uint8, np.uint16): Cell-cell
            junction image.
        seg_img (np.array, shape=(n,n)): Binarized cell-cell junction image.
        window_size (int, optional): Size of the kernel (both in x and y).
            Defaults to 7.

    Returns:
        np.array, shape=(n,n), dtype=float: Resulting image, where each point
            has the result of the application of the filter.
    """
    filtered = window_stdev(ccj_img, window_size=window_size)
    mask_filtered = mask_results(filtered, seg_img)
    return mask_filtered

def texture_entropy_filter(ccj_img, seg_img, window_size=7):
    """Applies a texture filter over ccj_img, and then uses seg_img to mask those
    results. The texture is computed using an entropy filter.

    Args:
        ccj_img (np.array, shape=(n,n), dtype=np.uint8): Cell-cell junction image.
        seg_img (np.array, shape=(n,n)): Binarized cell-cell junction image.
        window_size (int, optional): Size of the kernel (both in x and y).
            Defaults to 7.

    Returns:
        np.array, shape=(n,n), dtype=float: Resulting image, where each point
            has the result of the application of the filter.
    """
    if ccj_img.dtype == np.uint16:
        ccj_img = utils.convert_16_to_8(ccj_img)
    filtered = entropy(ccj_img, np.ones((window_size, window_size)))
    mask_filtered = mask_results(filtered, seg_img)
    return mask_filtered

def skeleton_data(sk_img):
    """Computes the skeleton data from a skeletonized image. For more
    information, go to https://jni.github.io/skan/getting_started.html .

    Args:
        sk_image (np.array, shape=(n,n)): Binarized skeleton image.

    Returns:
        scipy.sparse.csr.csr_matrix: Matrix in which entry (i,j) is 0 if pixels
            i and j are not connected, and otherwise is equal to the distance
            between pixels i and j in the skeleton.
        np.ndarray, shape=(n_points, 2): Coordinates of each point in the
            skeleton.
        np.ndarray, shape=(n, n): image of the skeleton, with each skeleton
            pixel containing the number of neighbouring pixel.
        pd.core.frame.DataFrame: dataframe with the summarized information.
    """
    pixel_graph, coordinates, degrees = skeleton_to_csgraph(sk_img)
    branch_data = summarize(Skeleton(sk_img))
    return pixel_graph, coordinates, degrees, branch_data
