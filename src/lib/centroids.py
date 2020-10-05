import cv2
import numpy as np
from skimage import filters
from skimage.draw import line
from scipy.spatial import Delaunay, cKDTree, Voronoi,  voronoi_plot_2d


def clip_image(img, new_size):
    # It is assumed that there is a uniform padding around the image, so the
    # returned image is clipped at the center.
    height, width = img.shape[:2]
    new_height, new_width = new_size
    i_coord = (height - new_height) // 2
    j_coord = (width - new_width) // 2
    return img[i_coord: i_coord + new_height, j_coord: j_coord + new_width]


def watershed_segmentation(original_img, bin_img):
    """Performs image segmentation using the watershed algorithm, according to
    https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html.

    Args:
        original_img: Original image, 3-channels.
        bin_img (np.ndarray): Binary image, 1-channel, 8-bit depth (i.e., all
            values are 0 or 255)

    Returns:
        list(np.ndarray): list of contours. Each element on the list is an
            array of coordinates which represents a contour.
    """
    # Noise removal
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform,
                                0.3 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers_watershed = cv2.watershed(original_img, markers)

    contours_img = np.zeros(markers_watershed.shape, dtype='uint8')
    contours_img[markers_watershed == -1] = 255

    contours, _ = cv2.findContours(contours_img, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)

    return contours

def get_nuclei_centroids(nuclei_img):
    """Given a nuclei image, returns a list with the centroid of each nucleus.

    Args:
        nuclei_img (np.array): Nuclei image, in BGR format.

    Returns:
        np.ndarray: Array with the coordinates of the centroids of the nuclei.
            Before returning, the list of centroids is filtered, in order to}
            avoid duplicates.
    """

    pixels_border = 20

    nuclei_img_expanded = cv2.copyMakeBorder(nuclei_img,
                                             pixels_border, pixels_border,
                                             pixels_border, pixels_border,
                                             cv2.BORDER_CONSTANT, 0)

    nuclei_img_gray = cv2.cvtColor(nuclei_img_expanded, cv2.COLOR_BGR2GRAY)

    # Enhance contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)

    clahe = cv2.createCLAHE(clipLimit=5.0)
    contrast_img = clahe.apply(nuclei_img_gray)

    # Blurring
    blur_img = cv2.GaussianBlur(contrast_img, (9,9),0)

    # Thresholding (Otsu)
    _, thresh_img = cv2.threshold(blur_img, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours = watershed_segmentation(nuclei_img_expanded, thresh_img)
    contours = contours[:-2]

    centroids = []
    for contour in contours:
        moments = cv2.moments(contour)
        area = moments['m00']
        x_coord = int(moments['m10'] / area) - pixels_border
        y_coord = int(moments['m01'] / area) - pixels_border
        centroids.append((x_coord, y_coord))

    centroids = np.array(centroids, dtype=int)

    return np.unique(centroids, axis=0)

def merge_centroids(centroids, rows_to_fuse):
    """Given a list of centroids and a list of pairs of rows, merges those rows
    in the list of centroids.

    Args:
        centroids (np.ndarray, shape=(n_centroids, 2)): Coordinates of nuclei
            centroids.
        rows_to_fuse (np.ndarray, shape=(n_merge, 2)): List of pairs of
            centroids that should be merged. Each point in rows_to_fuse is a
            pair of indexes of rows in centroids that should be merged.

    Returns:
        np.ndarray, shape=(n_centroids - n_merge, 2): Coordinates of nuclei
            centroids after merging.
    """
    if len(rows_to_fuse) == 0:
        return centroids
    for rows in rows_to_fuse:
        centroids[rows[0]] = (centroids[rows[0]] + centroids[rows[1]]) / 2

    mask = np.ones(len(centroids), dtype=bool)
    mask[rows_to_fuse[:, 1]] = False
    return centroids[mask]

def remove_close_centroids(centroids, radio=10):
    """Remove those centroids that are too close to each other.

    Args:
        centroids (np.ndarray, shape=(n_centroids, 2)): Coordinates of nuclei
            centroids.
        radio (int, optional): Minimum distance required between each pair of
            centroids. Defaults to 10.

    Returns:
        np.ndarray: List of coordinates after removal of too close centroids.
    """
    kdtree = cKDTree(centroids)
    rows_to_fuse = np.array(list(kdtree.query_pairs(r=radio)))
    merged_centroids = merge_centroids(centroids, rows_to_fuse)
    return merged_centroids

def get_edges_indexes_delaunay(delaunay_tri):
    """Gets all the unique edges in the Delaunay triangulation.

    Returns the indexes of the points that forms each edge. The indexes are
    relative to the list of points stored in the Delaunay triangulation.

    Args:
        delaunay_tri (scipy.spatial.qhull.Delaunay): Delaunay triangulation.

    Returns:
        np.ndarray, shape=(n_edges,2): List of unique edges in the
            triangulation.
    """
    # We get the references to the points that forms every edge
    edges_refs = []
    for simplex in delaunay_tri.simplices:
        edge_ref_1 = np.array((simplex[0], simplex[1]))
        edge_ref_2 = np.array((simplex[1], simplex[2]))
        edge_ref_3 = np.array((simplex[2], simplex[0]))

        # We sort the points, so that we can later delete the duplicates
        edge_ref_1.sort()
        edge_ref_2.sort()
        edge_ref_3.sort()

        edges_refs.append(edge_ref_1)
        edges_refs.append(edge_ref_2)
        edges_refs.append(edge_ref_3)

    # Remove the duplicates
    edges_refs = np.unique(edges_refs, axis=0)
    return edges_refs

def get_indexes_false_centroids(segments, edges_refs, bin_img):
    """Returns the indexes of those centroids that have at least one adjacent
    centroid with no cell-cell junction in between.

    Each segment joins two adjacent nuclei centroids. So for each segment, we
    check that it contains at least one white pixel. If it doesn't, it means
    that there is no cell-cell junction between the two centroids, and then
    those two centroids should be merged into one.

    Args:
        segments (list[(np.ndarray, np.ndarray)], length=n_edges): List of
            segments joining two adjacent centroids. Each segment is represented
            by 2 numpy arrays of equal length, namely rows_idx and cols_idx,
            such that bin_img[rows_idx, cols_idx] returns the pixels that
            belongs to the segment.
        edges_refs (np.ndarray, shape=(n_edges,2)): List of unique edges in the
            triangulation.
        bin_img (np.ndarray):  Binary image, 3-channel, 8-bit depth (i.e., all
            values are 0 or 255)

    Returns:
        np.ndarray, shape=(n_merge, 2). List of pairs of centroids indexes that
            should be merged because there is no cell-cell junction between
            them.
    """

    false_centroids_indexes = []
    for index, segment in enumerate(segments):
        rows_idx, cols_idx = segment
        segment_img = bin_img[cols_idx, rows_idx]
        if not np.any(segment_img == 255):
            false_centroids_indexes.append(edges_refs[index])
    return np.array(false_centroids_indexes)


def clean_centroids_delaunay(centroids, bin_img):
    """Given a list of centroids that represents the nuclei and a segmented
    image of the cell-cell junction, removes those centroids that belong to the
    same cell.

    To do that, we compute the Delaunay triangulation over the centroids, and
    then we merge those pairs of centroids where there is no cell-cell junction
    between them.

    Args:
        centroids (np.ndarray, shape=(n_centroids, 2)): Coordinates of nuclei
            centroids.
        bin_img (np.ndarray):  Binary image, 3-channel, 8-bit depth (i.e., all
            values are 0 or 255)

    Returns:
        np.ndarray, shape=(<=n_centroids, 2): Coordinates of nuclei
            centroids after merging the corresponding points.
    """
    # Delaunay triangulation
    tri = Delaunay(centroids)
    edges_refs = get_edges_indexes_delaunay(tri)

    # Now we get the real coords that forms every edge, not just the references
    edges = centroids[edges_refs]

    # We get the actual segments. Not just the two points, but the whole line
    interp_segments = []
    for edge in edges:
        rows, columns = line(edge[0, 0], edge[0, 1], edge[1, 0], edge[1, 1])
        interp_segments.append((rows, columns))
    false_centroids_indexes = get_indexes_false_centroids(interp_segments,
                                                          edges_refs, bin_img)
    merged_centroids = merge_centroids(centroids, false_centroids_indexes)
    return merged_centroids


def remove_white_centroids(centroids, bin_img):
    """Remove those centroids that are in cell-cell junction, according to
    bin_img.

    Args:
        centroids (np.ndarray, shape=(n_centroids, 2)): Coordinates of
            nuclei centroids.
        bin_img (np.ndarray):  Binary image, 3-channel, 8-bit depth (i.e., all
            values are 0 or 255)

    Returns:
        np.ndarray, shape=(<=n_centroids, 2): Coordinates of nuclei
            centroids after merging the corresponding points.
    """
    mask = np.ones(len(centroids), dtype=bool)
    for index, centroid in enumerate(centroids):
        if np.all(bin_img[centroid[1], centroid[0]] == (255, 255, 255)):
            mask[index] = False
    cleaned_centroids = centroids[mask]
    return cleaned_centroids


def clean_centroids(centroids, bin_img, n_iter=3, radio=10):
    """Clean the list of centroids, removing those who

    Args:
        centroids ([type]): [description]
        bin_img ([type]): [description]
        n_iter (int, optional): [description]. Defaults to 3.
        radio (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    for index in range(n_iter):
        print("########## Iteration {} ########## ".format(index+1))
        print("N centroids: {}".format(len(centroids)))
        centroids = remove_white_centroids(centroids, bin_img)
        print("N centroids after removing centroids in white space: {}".format(len(centroids)))
        centroids = remove_close_centroids(centroids, radio)
        print("N centroids after removing close points: {}".format(len(centroids)))
        centroids = clean_centroids_delaunay(centroids, bin_img)
        print("N centroids after Delaunay analysis: {}".format(len(centroids)))
    return centroids


def get_moments(contours):
    moments = []
    for c in contours:
        m = cv2.moments(c)
        moments.append(m)
    return moments

def get_colored_contours(contours, img_template):
    colors = np.random.randint(0, 256, (len(contours), 3)).astype('uint8')
    colors_dict = {tuple(colors[index]): index for index in range(len(colors))}

    contours_color = np.zeros_like(img_template)
    for contour, color in zip(contours, colors):
        color_t = tuple(int(c) for c in color)
        cv2.fillPoly(contours_color, [contour], color=color_t)
    return contours_color, colors_dict

def get_moments_centroids(centroids, contours, img_template):
    moments = get_moments(contours)
    ret = []
    colored_img, colors_dict = get_colored_contours(contours, img_template)
    for centroid in centroids:
        color = tuple(colored_img[centroid[1], centroid[0]])
        index = colors_dict[color]
        ret.append(moments[index])
    return ret

def remove_outliers_cells(centroids, moments, std_factor=2.5):
    """Given a list of centroids and its moments, removes those centroids whose
    area too big.

    In order to perform this task, we compute the mean and std of all the areas,
    and then we remove those centroids where (area - mean) > std_factor*std.

    Args:
        centroids (np.ndarray, shape=(n_centroids, 2)): Coordinates of nuclei
            centroids.
        moments (list[dict[str -> float]]): list of moments for each nucleus.
            Each element in the list is a dict with the moments of the
            corresponding nucleus, as returned by cv2.moments(nucleus_contour).
        std_factor (float, optional): Tolerance to outliers. Defaults to 2.5.

    Returns:
        (np.ndarray, list[dict[str -> float]]): filtered centroids and moments
    """
    moments_np = np.array(moments)
    areas = np.array([m['m00'] for m in moments])
    mask = np.ones(len(centroids), dtype=bool)
    mean, std = areas.mean(), areas.std()
    for index, (_, area) in enumerate(zip(centroids, areas)):
        if abs(area - mean) > std_factor*std:
            mask[index] = False
    return centroids[mask], moments_np[mask]

def get_moments_cells(centroids, bin_img, remove_outliers=True):
    inv_img = cv2.bitwise_not(bin_img)
    inv_img = cv2.cvtColor(inv_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(inv_img, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)
    moments = get_moments_centroids(centroids, contours, bin_img)
    if remove_outliers:
        return remove_outliers_cells(centroids, moments)
    else:
        return centroids, moments
