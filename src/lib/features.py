import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage.filters import uniform_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters.rank import entropy
from skimage.morphology import medial_axis
from skan import skeleton_to_csgraph, summarize, Skeleton
from . import centroids as centutils
from . import utils

def mask_results(results_img, jbin_img):
    """Applies a mask over the results of a operation.

    Args:
        results_img (np.array, shape=(n,n)): Image with the results of an
            operation.
        jbin_img (np.array, shape=(n,n)): Mask.

    Returns:
        [type]: [description]
    """
    results_copy = results_img.copy()
    results_copy[jbin_img == 0] = 0
    return results_copy

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


def branch_thickness_voronoi(img):
    """Computes the thickness of the branches using an approximation of the
    generalized Voronoi algorithm.

    Args:
        img (np.array, shape=(n,n)): Binarized cell-cell junction image.

    Returns:
        np.array, shape=(n,n), dtype=float: Resulting image, where each point
            of the skeleton has the thickness in that point.
    """
    distance = ndi.distance_transform_edt(img)
    local_maxi = peak_local_max(-distance, indices=False,
                                footprint=np.ones((3, 3)), labels=img)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(distance, markers, watershed_line=True)
    mask = labels == 0
    distance_on_skel = distance.copy()
    distance_on_skel[~mask] = 0
    return distance_on_skel

def branch_thickness_medial_axis(img):
    """Computes the thickness of the branches using the medial axis algorithm.

    Args:
        img (np.array, shape=(n,n)): Binarized cell-cell junction image.

    Returns:
        np.array, shape=(n,n), dtype=float: Resulting image, where each point
            of the skeleton has the thickness in that point.
    """
    skel, distance = medial_axis(img, return_distance=True)
    dist_on_skel = distance * skel
    return dist_on_skel

def texture_std_filter(img, jbin_img, window_size=7):
    """Applies a texture filter over img, and then uses jbin_img to mask those
    results. The texture is computed using a standard deviation filter.

    Args:
        img (np.array, shape=(n,n)): Cell-cell junction image.
        jbin_img (np.array, shape=(n,n)): Binarized cell-cell junction image.
        window_size (int, optional): Size of the kernel (both in x and y).
            Defaults to 7.

    Returns:
        np.array, shape=(n,n), dtype=float: Resulting image, where each point
            has the result of the application of the filter.
    """
    filtered = window_stdev(img, window_size=window_size)
    mask_filtered = mask_results(filtered, jbin_img)
    return mask_filtered

def texture_entropy_filter(img, jbin_img, window_size=7):
    """Applies a texture filter over img, and then uses jbin_img to mask those
    results. The texture is computed using an entropy filter.

    Args:
        img (np.array, shape=(n,n)): Cell-cell junction image.
        jbin_img (np.array, shape=(n,n)): Binarized cell-cell junction image.
        window_size (int, optional): Size of the kernel (both in x and y).
            Defaults to 7.

    Returns:
        np.array, shape=(n,n), dtype=float: Resulting image, where each point
            has the result of the application of the filter.
    """
    filtered = entropy(img, np.ones((window_size, window_size)))
    mask_filtered = mask_results(filtered, jbin_img)
    return mask_filtered

def skeleton_data(sk_img):
    """Computes the skeleton data from a skeletonized image. For more
    information, see https://jni.github.io/skan/getting_started.html#extracting-a-skeleton-from-an-image .

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


def get_statistics_from_processed_image(results_img):
    non_zero = results_img[np.where(results_img != 0)]
    return non_zero.mean(), non_zero.std()


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


def get_skeleton_features(sk_img):
    degrees, branch_data = skeleton_data(sk_img)

    ret = {}

    # End-to-end
    e2e_data = branch_data[branch_data['branch-type'] == 0]
    ret['e2e_n'] = len(e2e_data)
    ret['e2e_distance_mean'] = e2e_data['branch-distance'].mean()
    ret['e2e_distance_std'] = e2e_data['branch-distance'].std()
    ret['e2e_eu_distance_mean'] = e2e_data['euclidean-distance'].mean()
    ret['e2e_eu_distance_std'] = e2e_data['euclidean-distance'].std()
    ret['e2e_distance_ratio_mean'] = \
        (e2e_data['euclidean-distance'] / e2e_data['branch-distance']).mean()
    ret['e2e_distance_ratio_std'] = \
         (e2e_data['euclidean-distance'] / e2e_data['branch-distance']).std()

    # Junction-to-end
    j2e_data = branch_data[branch_data['branch-type'] == 1]
    ret['j2e_n'] = len(j2e_data)
    ret['j2e_distance_mean'] = j2e_data['branch-distance'].mean()
    ret['j2e_distance_std'] = j2e_data['branch-distance'].std()
    ret['j2e_eu_distance_mean'] = j2e_data['euclidean-distance'].mean()
    ret['j2e_eu_distance_std'] = j2e_data['euclidean-distance'].std()
    ret['j2e_distance_ratio_mean'] = \
        (j2e_data['euclidean-distance'] / j2e_data['branch-distance']).mean()
    ret['j2e_distance_ratio_std'] = \
         (j2e_data['euclidean-distance'] / j2e_data['branch-distance']).std()

    # Junction-to-junction
    j2j_data = branch_data[branch_data['branch-type'] == 2]
    ret['j2j_n'] = len(j2j_data)
    ret['j2j_distance_mean'] = j2j_data['branch-distance'].mean()
    ret['j2j_distance_std'] = j2j_data['branch-distance'].std()
    ret['j2j_eu_distance_mean'] = j2j_data['euclidean-distance'].mean()
    ret['j2j_eu_distance_std'] = j2j_data['euclidean-distance'].std()
    ret['j2j_distance_ratio_mean'] = \
        (j2j_data['euclidean-distance'] / j2j_data['branch-distance']).mean()
    ret['j2j_distance_ratio_std'] = \
        (j2j_data['euclidean-distance'] / j2j_data['branch-distance']).std()

    # Nodes
    nodes = degrees[np.where(degrees > 2)]
    ret['nodes_n'] = len(nodes)
    ret['nodes_max'] = nodes.max()
    ret['nodes_mean'] = nodes.mean()
    ret['nodes_std'] = nodes.std()

    return ret

def get_branch_thickness_features(jbin_img):
    ret = {}

    bt_medial_axis = branch_thickness_medial_axis(jbin_img)
    bt_voronoi = branch_thickness_voronoi(jbin_img)

    ma_mean, ma_std = get_statistics_from_processed_image(bt_medial_axis)
    va_mean, va_std = get_statistics_from_processed_image(bt_voronoi)

    ret['medial_axis_mean'] = ma_mean
    ret['medial_axis_std'] = ma_std
    ret['voronoi_approx_mean'] = va_mean
    ret['voronoi_approx_std'] = va_std

    return ret

def get_texture_features(original_img, jbin_img):
    ret = {}
    # Entropy filter is too slow when using 16-bit depth images, so we need
    # to convert it to 8-bit depth.
    original_img_8 = utils.convert_16_to_8(original_img)

    tx_std = texture_std_filter(original_img, jbin_img)
    tx_entropy = texture_entropy_filter(original_img_8, jbin_img)

    sf_mean, sf_std = get_statistics_from_processed_image(tx_std)
    ef_mean, ef_std = get_statistics_from_processed_image(tx_entropy)

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
    nuclei_img_bgr = utils.convert_16_gray_to_8_bgr(nuclei_img)
    jbin_img_bgr = cv2.cvtColor(jbin_img, cv2.COLOR_GRAY2BGR)

    area_ft = get_area_features(nuclei_img_bgr, jbin_img)
    sk_ft = get_skeleton_features(jsk_img)
    bt_ft = get_branch_thickness_features(jbin_img)
    tx_ft = get_texture_features(original_img, jbin_img)

    features.update(area_ft)
    features.update(sk_ft)
    features.update(bt_ft)
    features.update(tx_ft)

    return features