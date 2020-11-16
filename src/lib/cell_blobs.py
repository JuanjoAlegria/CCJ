import cv2
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d


def get_contours_and_moments(seg_img, std_factor=3):
    seg_inv = cv2.bitwise_not(seg_img)
    contours, _ = cv2.findContours(seg_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    moments = []
    for cnt in contours:
        moment = cv2.moments(cnt)
        moments.append(moment)
    areas = np.array([m['m00'] for m in moments])
    areas_mean, areas_std = areas.mean(), areas.std()

    keep_indexes = []
    for index, moment in enumerate(moments):
        current_area = moment['m00']
        if current_area == 0 or \
            abs(current_area - areas_mean) > std_factor * areas_std:
            continue
        keep_indexes.append(index)

    contours = np.array(contours)[keep_indexes]
    moments = np.array(moments)[keep_indexes]
    return contours, moments

def get_region_measurements(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    compactness = 4*np.pi*area / (perimeter**2)
    solidity = area / hull_area
    convexity = hull_perimeter / perimeter
    min_bbox = cv2.minAreaRect(cnt)
    major_axis_length = max(min_bbox[1])
    minor_axis_length = min(min_bbox[1])
    elongation = minor_axis_length / major_axis_length
    res = {
        'area': area,
        'perimeter': perimeter,
        'hull_area': hull_area,
        'hull_perimeter': hull_perimeter,
        'compactness': compactness,
        'solidity': solidity,
        'convexity': convexity,
        'major_axis_length': major_axis_length,
        'minor_axis_length': minor_axis_length,
        'elongation': elongation
    }
    return res

def get_boundary_measurements(cnt):
    radial_distances = get_radial_dimension(cnt)
    xs = np.arange(len(radial_distances))
    fractal_dimension = get_fractal_dimension_1d(xs, radial_distances)
    ent = get_entropy(radial_distances)
    res = {
        'fractal_dimension': fractal_dimension,
        'entropy': ent
    }
    return res

def get_entropy(ys):
    hist, _ = np.histogram(ys, bins='fd')
    hist_norm = hist / hist.sum()
    return stats.entropy(hist_norm, base=2)

def get_fractal_dimension_1d(xs, ys, n_points=4096):
    def count_boxes_1d(xs, ys, k, k_max):
        range_y_max = ys.max() - ys.min()
        indexes_x = np.arange(0, len(xs), 2**k)
        ranges_y = np.linspace(0, range_y_max, 2**(k_max - k) + 1) + ys.min()
        S = []
        # Optimization: we check for all the ys if they are in some y-range
        # at the same time, and then we collapse the result into boxes
        # (yeah, I feel proud about this trick)
        for j in range(len(ranges_y) - 1):
            r = (ys >= ranges_y[j]) & (ys <= ranges_y[j+1])
            S.append(np.add.reduceat(r, indexes_x))
        boxes = np.vstack(S)
        # print(boxes) # With this you can see the formation of the plot!
        return len(np.where(boxes > 0)[0])

    f_interp = interp1d(xs, ys)
    xs_new = np.linspace(0, len(xs) - 1, n_points, endpoint=True)
    ys_new = f_interp(xs_new)

    k_max = int(np.floor(np.log2(n_points)))
    ks = np.arange(k_max, 1, -1)
    sizes = 2**ks

    counts = []
    for k in ks:
        counts.append(count_boxes_1d(xs_new, ys_new, k, k_max))

    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    return coeffs[0]

def get_fractal_dimension_2d(img):
    """Computes the fractal dimension of a binary image. Adapted from
    https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1, using
    the Minkowski-Bouligand dimension
    (https://en.wikipedia.org/wiki/Minkowski%E2%80%93Bouligand_dimension).


    Args:
        img (np.array, dtype=uint8, ndim=2): Binary image over which the fractal
            dimension is going to be computed.

    Returns:
        float: fractal dimension.
    """

    # Only for 2d image
    assert(len(img.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
                               np.arange(0, img.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    img = img.astype(bool)

    # Minimal dimension of image
    p = min(img.shape)

    # Greatest power of 2 such as 2**n <= p
    n = int(np.floor(np.log2(p)))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(img, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def get_radial_dimension(cnt, moment=None, normalize=True):
    if moment is None:
        moment = cv2.moments(cnt)
    cx = int(moment['m10']/moment['m00'])
    cy = int(moment['m01']/moment['m00'])

    cnt_reshaped = cnt.reshape(len(cnt), 2)
    rd = np.linalg.norm(cnt_reshaped - [cx, cy], axis=1)
    if normalize:
        rd /= rd.max()
    return rd


def get_blobs_measurements(seg_img):
    seg_inv = cv2.bitwise_not(seg_img)
    contours, moments = get_contours_and_moments(seg_inv)
    merged = []
    for cnt, mnt in zip(contours, moments):
        current_dict = {
            'x': int(mnt['m10']/mnt['m00']),
            'y': int(mnt['m01']/mnt['m00'])
        }
        reg_ms = get_region_measurements(cnt)
        bry_ms = get_boundary_measurements(cnt)

        current_dict.update(mnt)
        current_dict.update(reg_ms)
        current_dict.update(bry_ms)
        merged.append(current_dict)
    df = pd.DataFrame(merged)
    return df
