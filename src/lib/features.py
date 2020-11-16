"""Module with functions to summarize the features of an image"""

import numpy as np
from scipy import stats

def get_statistics_from_processed_image(processed_img):
    """Utilitary function that computes the mean, std, median and median
    absolute deviation (mad) for the non-zero pixels of a processed image"

    Args:
        processed_img (np.ndarray): Processed image

    Returns:
        dict[str->float]: Dictionary with the computed statistics, with the keys
        'mean', 'std', 'median' and 'mad'.
    """
    non_zero = processed_img[np.where(processed_img != 0)]
    statistics = {
        'mean': non_zero.mean(),
        'std':  non_zero.std(),
        'median': np.median(non_zero),
        'mad': stats.median_absolute_deviation(non_zero, scale=1, axis=None)
    }
    return statistics

def get_area_features(seg_img, centroids_data):
    """Computes the area features of an image.

    The features are:

        - cell_area_ratio: Sum of cell areas divided by the total area of the
            image.
        - white_area_ratio: Sum of white pixels in the segmented image divided
            by the total area of the image.

    Args:
        seg_img (np.ndarray): Segmented cell cell junction image.
        centroids_data (pd.core.frame.DataFrame): Dataframe with the information
            of the centroids and moments.

    Returns:
        dict: Dictionary with the area features.
    """
    ret = {}
    cell_area = centroids_data['m00'].sum()
    white_area = len(seg_img[seg_img == 255])
    total_area = seg_img.shape[0] * seg_img.shape[1]
    ret['cell_area_ratio'] = cell_area / total_area
    ret['white_area_ratio'] = white_area / total_area
    return ret

def get_blobs_features(blobs_data):
    """Computes the blobs features of an image.

    For each feature we compute the mean, standard deviation, median and
    median absolute deviation. Those features types are:
        area: Area of the blob.
        perimeter: Perimeter of the blob.
        hull_area: Area of the convex hull.
        hull_perimeter: Perimeter of the convex hull.
        compactness: Ratio of the area of an object to the area of a circle
            with the same perimeter (4*pi*area / perimeter**2).
        solidity: Measures the density of an object (area / hull_area).
        convexity: Relative amount that an object differs from a convex object
            (hull_perimeter / perimeter).
        major_axis_length: Length of the major axis.
        minor_axis_length: Length of the minir axis.
        elongation: Ratio between the length of the axes
            (minor_axis_length / major_axis_length).
        fractal_dimension: Fractal dimension of the boundary of the blob,
            computed using the box-count estimator.
        entropy: Entropy of the boundary of the blob.

    Args:
        blobs_data (pd.core.frame.DataFrame): Dataframe with the information
            of the blobs.

    Returns:
        dict: Dictionary with the area features.
    """
    ret = {}
    features_names = ['area', 'perimeter', 'hull_area', 'hull_perimeter',
                      'compactness', 'solidity', 'convexity',
                      'major_axis_length', 'minor_axis_length', 'elongation',
                      'fractal_dimension', 'entropy']

    for ft_name in features_names:
        column = blobs_data[ft_name]
        ret[f'{ft_name}_mean'] = column.mean()
        ret[f'{ft_name}_std'] = column.std()
        ret[f'{ft_name}_median'] = np.median(column)
        ret[f'{ft_name}_mad'] = stats.median_absolute_deviation(column, scale=1)
    return ret

def get_skeleton_features(skeleton_data):
    """Computes the skeleton features of an image.

    The features are divided according to the type of branch, as follows:

    Endpoint-to-endpoint branches:
        - e2e_n: Number of branches of this type.
        - e2e_distance_mean: Mean of the natural distance of the branches of
            this type. This distances are measured as the sum of the pixels
            along the branch.
        - e2e_distance_std: Standard deviation of the previous measure.
        - e2e_eu_distance_mean: Mean of the Euclidean distance of the branches
            of this type. This distances are measured as the Euclidean distance
            between the starting point and the ending point.
        - e2e_eu_distance_std: Standard deviation of the previous measure.
        - e2e_distance_ratio_mean: For each branch, we compute the Euclidean
            distance divided by the natural distance, and then we compute the
             mean of these values.
        - e2e_distance_ratio_std: Standard deviation of the previous measure.

    Junction-to-endpoint:
        - j2e_n
        - j2e_distance_mean
        - j2e_distance_std
        - j2e_eu_distance_mean
        - j2e_eu_distance_std
        - j2e_distance_ratio_mean
        - j2e_distance_ratio_std

    Junction-to-junction:
        - j2j_n
        - j2j_distance_mean
        - j2j_distance_std
        - j2j_eu_distance_mean
        - j2j_eu_distance_std
        - j2j_distance_ratio_mean
        - j2j_distance_ratio_std

    Args:
        skeleton_data (pd.core.frame.DataFrame): Dataframe with the information
            of the skeleton.

    Returns:
        dict: Dictionary with the skeleton features.
    """
    ret = {}

    # End-to-end
    e2e_data = skeleton_data[skeleton_data['branch-type'] == 0]
    e2e_ratio_distances = e2e_data['euclidean-distance'] /  \
                            e2e_data['branch-distance']
    ret['e2e_n'] = len(e2e_data)

    ret['e2e_distance_mean'] = e2e_data['branch-distance'].mean()
    ret['e2e_distance_std'] = e2e_data['branch-distance'].std()
    ret['e2e_distance_median'] = np.median(e2e_data['branch-distance'])
    ret['e2e_distance_mad'] = stats.median_absolute_deviation(
                                        e2e_data['branch-distance'], scale=1)

    ret['e2e_eu_distance_mean'] = e2e_data['euclidean-distance'].mean()
    ret['e2e_eu_distance_std'] = e2e_data['euclidean-distance'].std()
    ret['e2e_eu_distance_median'] = np.median(e2e_data['euclidean-distance'])
    ret['e2e_eu_distance_mad'] = stats.median_absolute_deviation(
                                        e2e_data['euclidean-distance'], scale=1)


    ret['e2e_distance_ratio_mean'] = e2e_ratio_distances.mean()
    ret['e2e_distance_ratio_std'] = e2e_ratio_distances.std()
    ret['e2e_distance_ratio_median'] = np.median(e2e_ratio_distances)
    ret['e2e_distance_ratio_mad'] = stats.median_absolute_deviation(
                                        e2e_ratio_distances, scale=1)


    # Junction-to-end
    j2e_data = skeleton_data[skeleton_data['branch-type'] == 1]
    j2e_ratio_distances = j2e_data['euclidean-distance'] /  \
                            j2e_data['branch-distance']
    ret['j2e_n'] = len(j2e_data)

    ret['j2e_distance_mean'] = j2e_data['branch-distance'].mean()
    ret['j2e_distance_std'] = j2e_data['branch-distance'].std()
    ret['j2e_distance_median'] = np.median(j2e_data['branch-distance'])
    ret['j2e_distance_mad'] = stats.median_absolute_deviation(
                                        j2e_data['branch-distance'], scale=1)

    ret['j2e_eu_distance_mean'] = j2e_data['euclidean-distance'].mean()
    ret['j2e_eu_distance_std'] = j2e_data['euclidean-distance'].std()
    ret['j2e_eu_distance_median'] = np.median(j2e_data['euclidean-distance'])
    ret['j2e_eu_distance_mad'] = stats.median_absolute_deviation(
                                        j2e_data['euclidean-distance'], scale=1)


    ret['j2e_distance_ratio_mean'] = j2e_ratio_distances.mean()
    ret['j2e_distance_ratio_std'] = j2e_ratio_distances.std()
    ret['j2e_distance_ratio_median'] = np.median(j2e_ratio_distances)
    ret['j2e_distance_ratio_mad'] = stats.median_absolute_deviation(
                                        j2e_ratio_distances, scale=1)


    # Junction-to-junction
    j2j_data = skeleton_data[skeleton_data['branch-type'] == 2]
    j2j_ratio_distances = j2j_data['euclidean-distance'] /  \
                            j2j_data['branch-distance']
    ret['j2j_n'] = len(j2j_data)

    ret['j2j_distance_mean'] = j2j_data['branch-distance'].mean()
    ret['j2j_distance_std'] = j2j_data['branch-distance'].std()
    ret['j2j_distance_median'] = np.median(j2j_data['branch-distance'])
    ret['j2j_distance_mad'] = stats.median_absolute_deviation(
                                        j2j_data['branch-distance'], scale=1)

    ret['j2j_eu_distance_mean'] = j2j_data['euclidean-distance'].mean()
    ret['j2j_eu_distance_std'] = j2j_data['euclidean-distance'].std()
    ret['j2j_eu_distance_median'] = np.median(j2j_data['euclidean-distance'])
    ret['j2j_eu_distance_mad'] = stats.median_absolute_deviation(
                                        j2j_data['euclidean-distance'], scale=1)


    ret['j2j_distance_ratio_mean'] = j2j_ratio_distances.mean()
    ret['j2j_distance_ratio_std'] = j2j_ratio_distances.std()
    ret['j2j_distance_ratio_median'] = np.median(j2j_ratio_distances)
    ret['j2j_distance_ratio_mad'] = stats.median_absolute_deviation(
                                        j2j_ratio_distances, scale=1)

    return ret

def get_nodes_degrees_features(degrees_img):
    """Computes the nodes degrees features of an image.

    The skeleton can be thinked of as a graph, and, in this case, a node is
    defined as a point whose degree is greater than two. So, the computed
    features are:

        - nodes_n: Number of nodes in the graph.
        - nodes_max: Maximum degree in the graph
        - nodes_mean: Mean of the degrees in the graph
        - nodes_std: Standard deviation of the previous measure

    Args:
        degrees_img (np.ndarray): Image of the skeleton, with each skeleton
            pixel containing the number of neighbouring pixel.

    Returns:
        dict: Dictionary with the nodes degrees features.
    """
    ret = {}
    nodes = degrees_img[np.where(degrees_img > 2)]
    ret['nodes_n'] = len(nodes)
    ret['nodes_max'] = nodes.max()
    ret['nodes_mean'] = nodes.mean()
    ret['nodes_std'] = nodes.std()
    ret['nodes_median'] = np.median(nodes)
    ret['nodes_mad'] = stats.median_absolute_deviation(nodes, scale=1)
    return ret

def get_statistics_features(processed_img, feature_name):
    """Computes the mean and standard deviation for a generic processed image,
    where the only values that should be considered are the non-zero pixels.

    It is used to compute the branch thickness and texture features. In detail:

    Branch thickness features:
        - branch_thickness_medial_axis_mean: For each point in the skeleton, we
            compute its width in the corresponding segmented image, using the
            medial axis algorithm. And then we compute the mean of those values
            all over the image.
        - branch_thickness_medial_axis_std: Standard deviation of the previous
            measure.
        - branch_thickness_voronoi_mean: the same idea, but using an
            approximation of the Generalized Voronoi algorithm
        - branch_thickness_voronoi_mean_std: Standard deviation of the previous
            measure.

    Texture features:
        - texture_std_filter_mean: for each pixel in the cell-cell junction, we
            compute the standard deviation of the surrounding pixels. And then
            we compute the mean of those values all over the image.
        - texture_std_filter_std: Standard deviation of the previous measure.
        - texture_entropy_filter_mean: The same idea, but measuring the entropy
            instead of the standard deviation.
        - texture_entropy_filter_std: Standard deviation of the previous
            measure.

    Args:
        processed_img (np.ndarray): Processed image.
        feature_name (string): Name of the feature we are computing.

    Returns:
        dict: Dictionary with the mean and std of the feature we are computing.
    """
    ret = {}
    dict_statistics = get_statistics_from_processed_image(processed_img)
    for st_name in ['mean', 'std', 'median', 'mad']:
        ret[f'{feature_name}_{st_name}'] = dict_statistics[st_name]
    return ret

def get_features(seg_img, centroids_data, blobs_data, skeleton_data,
                 degrees_img, bt_medial_axis_img, bt_voronoi_img,
                 tx_std_img, tx_entropy_img):
    """Utilitary function that gets all the features: area, skeleton, nodes
    degrees, branch thickness and texture features.

    Args:
        seg_img (np.ndarray): Segmented cell cell junction image.
        centroids_data (pd.core.frame.DataFrame): Dataframe with the information
            of the centroids and moments.
        blobs_data (pd.core.frame.DataFrame): Dataframe with the information
            of the blobs.
        skeleton_data (pd.core.frame.DataFrame): Dataframe with the information
            of the skeleton.
        degrees_img (np.ndarray): Image of the skeleton, with each skeleton
            pixel containing the number of neighbouring pixel.
        bt_medial_axis_img (np.ndarray): Processed image with the branch
            thickness of each point of the skeleton, using the medial axis
            algorithm.
        bt_voronoi_img (np.ndarray): Processed image with the branch
            thickness of each point of the skeleton, using an approximation of
            the generalized Voronoi algorithm.
        tx_std_img (np.ndarray): Processed image with the texture of each point
            of the cell cell junction, using a standard deviation filter.
        tx_entropy_img (np.ndarray): Processed image with the texture of each
            point of the cell cell junction, using an entropy filter.

    Returns:
        dict: Dictionary with all the features.
    """

    features = {}

    area_fts = get_area_features(seg_img, centroids_data)
    blobs_fts = get_blobs_features(blobs_data)
    sk_fts = get_skeleton_features(skeleton_data)
    dg_fts = get_nodes_degrees_features(degrees_img)

    features.update(area_fts)
    features.update(blobs_fts)
    features.update(sk_fts)
    features.update(dg_fts)

    map_names_imgs = {'branch_thickness_medial_axis': bt_medial_axis_img,
                      'branch_thickness_voronoi': bt_voronoi_img,
                      'texture_std_filter': tx_std_img,
                      'texture_entropy_filter': tx_entropy_img}

    for ft_name, img in map_names_imgs.items():
        features.update(get_statistics_features(img, ft_name))

    return features
