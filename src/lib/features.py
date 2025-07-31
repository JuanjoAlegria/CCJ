"""Module with functions to summarize the features of an image"""

import numpy as np
from scipy import stats
from . import image_processing as iputils
from . import centroids as centutils
from . import cell_blobs_skimage as blobsutils


def compute_entropy(raw_vector):
    """Computes Shannon's entropy of a vector.

    Args:
        raw_vector (np.ndarray): Vector.

    Returns:
        float: Shannon's entropy.
    """

    _, counts = np.unique(raw_vector, return_counts=True)
    counts = counts / counts.sum()
    entropy = -np.sum(counts * np.log2(counts))
    return entropy


def compute_entropy_kde(vector, n_points=100):
    """Computes Shannon's entropy of a vector using a Kernel Density Estimator
    istead of the count of the elements.

    Args:
        raw_vector (np.ndarray): Vector.

    Returns:
        float: Shannon's entropy.
    """
    try:
        kernel = stats.gaussian_kde(vector, bw_method="scott")
        X = np.linspace(min(vector), max(vector), n_points)
        y = kernel(X)
        dx = X[1] - X[0]
        return -np.sum(y * np.log2(y) * dx)
    except Exception as e:
        print(e, vector)
        return -1


def compute_mode(vector):
    """Computes the mode of a vector.

    If the vector is formed only by integers, then we compute the mode just
    counting the numbers. Otherwise (the vector has floating point numbers), we
    compute the mode using an histogram.

    Args:
        vector (np.ndarray): Vector.

    Returns:
        int | float: mode.
    """
    if issubclass(vector.dtype.type, np.integer):
        return stats.mode(vector).mode[0]
    else:
        hist, bin_edges = np.histogram(vector, bins="auto")
        idx_max = hist.argmax()
        mode = (bin_edges[idx_max] + bin_edges[idx_max + 1]) / 2
        return mode


def get_summary_statistics(vector):
    """Utilitary function that computes the mean, std, median, median
    absolute deviation (mad), entropy and mode for a vector"

    Args:
        vector (np.ndarray, ndim=1): vector

    Returns:
        dict[str->float]: Dictionary with the computed statistics, with the keys
        'mean', 'std', 'median', 'mad' and 'entropy'.
    """
    if len(vector) == 0:
        mean = std = median = mad = entropy = mode = 0

    else:
        mean = vector.mean()
        std = vector.std()
        median = np.median(vector)
        mad = stats.median_absolute_deviation(vector, scale=1, axis=None)
        # entropy = compute_entropy_kde(vector)
        mode = compute_mode(vector)

    statistics = {
        "mean": mean,
        "std": std,
        "median": median,
        "mad": mad,
        # "entropy": entropy,
        "mode": mode,
    }
    return statistics


def get_area_features(seg_img):
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
    # cell_area = centroids_data["m00"].sum()
    # ret["cell_area_ratio"] = cell_area / total_area
    total_area = seg_img.shape[0] * seg_img.shape[1]
    white_area = len(seg_img[seg_img == 255])
    ret["white_area_ratio"] = white_area / total_area
    return ret


def get_blobs_features(blobs_data, min_area=900):
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
    features_names = [
        "area",
        "perimeter",
        "hull_area",
        "hull_perimeter",
        "compactness",
        "solidity",
        "convexity",
        "major_axis_length",
        "minor_axis_length",
        "elongation"
        # "fractal_dimension",
        # "entropy",
    ]

    blobs_data_no_borders = blobs_data[~blobs_data["is_at_border"]]
    dfs = [
        {"suffix": "", "df": blobs_data},
        {"suffix": "_borderless", "df": blobs_data_no_borders},
    ]

    if min_area is not None:
        blobs_data_no_small = blobs_data[blobs_data["area"] > min_area]
        dfs.append({"suffix": f"_>{min_area}", "df": blobs_data_no_small})

    for element in dfs:
        data, suffix = element["df"], element["suffix"]
        for ft_name in features_names:
            column = data[ft_name]
            sumstats = get_summary_statistics(column)
            ret.update(
                {f"{ft_name}{suffix}_{key}": value for key, value in sumstats.items()}
            )
        ret[f"blobs{suffix}_n"] = len(data)
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
    branches_types = {
        "e2e": 0,  # End-to-end
        "j2e": 1,  # Junction-to-end
        "j2j": 2,  # Junction-to-junction
    }

    for b_suffix, b_value in branches_types.items():
        data = skeleton_data[skeleton_data["branch-type"] == b_value]

        distances_types = {
            "distance": data["branch-distance"],
            "eu_distance": data["euclidean-distance"],
            "distance_ratio": data["euclidean-distance"] / data["branch-distance"],
        }

        ret[f"{b_suffix}_n"] = len(data)

        for dist_type, dist_data in distances_types.items():
            sumstats = get_summary_statistics(dist_data)
            ret.update(
                {
                    f"{b_suffix}_{dist_type}_{key}": value
                    for key, value in sumstats.items()
                }
            )
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
    if len(nodes) == 0:
        ret["nodes_n"] = 0
        ret["nodes_max"] = 0
        return ret

    ret["nodes_n"] = len(nodes)
    ret["nodes_max"] = nodes.max()
    sumstats = get_summary_statistics(nodes)
    ret.update({f"nodes_{key}": value for key, value in sumstats.items()})
    return ret


# TODO change name
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
    non_zero = processed_img[np.where(processed_img != 0)]
    sumstats = get_summary_statistics(non_zero)
    ret.update({f"{feature_name}_{key}": value for key, value in sumstats.items()})
    return ret


def get_global_dispersion_features(ccj_img, seg_img):
    ret = {}
    masked = iputils.get_mask(ccj_img, seg_img)
    ret["global_entropy_discrete"] = iputils.global_entropy_discrete(masked)
    ret["global_entropy_kde"] = iputils.global_entropy_kde(masked)
    ret["global_coeff_var"] = iputils.coefficient_variation(masked)
    return ret


def get_nuclei_blobs_features(
    blobs_data, total_area=2359296, min_area=2500, max_area=70000
):
    ret = {}
    # nuclei_centroids = centutils.remove_close_centroids(
    #     centutils.get_nuclei_centroids(nuclei_img), radio=50
    # )

    # h, w = nuclei_img.shape[:2]
    index_small = blobs_data["area"] <= min_area
    index_big = blobs_data["area"] >= max_area
    blobs_data_no_small = blobs_data[~index_small]
    blobs_data_small = blobs_data[index_small]
    blobs_data_big = blobs_data[index_big]

    total_area_small = blobs_data_small["area"].sum()
    total_area_big = blobs_data_big["area"].sum()

    # n_nuclei = len(nuclei_centroids)
    n_blobs = len(blobs_data)
    n_blobs_no_small = len(blobs_data_no_small)

    percentage_big = total_area_big / total_area
    percentage_small = total_area_small / total_area

    # ret["nuclei_n"] = n_nuclei
    # ret["nuclei_blobs_ratio"] = n_blobs_no_small / n_nuclei
    ret[f"blobs_<{min_area}_ratio"] = 1 - n_blobs_no_small / n_blobs
    ret[f"total_area_blobs<{min_area}"] = total_area_small
    ret[f"total_area_blobs>{max_area}"] = total_area_big
    ret[f"percentage_area_blobs<{min_area}"] = percentage_small
    ret[f"percentage_area_blobs>{max_area}"] = percentage_big

    return ret


def get_broken_features(broken_img, min_threshold=900):
    if broken_img is None:
        return {
            "area_broken": 0,
            "proportion_broken": 0,
            f"area_broken>{min_threshold}": 0,
            f"proportion_broken>{min_threshold}": 0,
        }
    ret = {}
    df_broken = blobsutils.get_blobs_measurements(
        broken_img, return_img=False, do_dilation=True
    )
    area_total = broken_img.shape[0] * broken_img.shape[1]
    area_broken = df_broken["area"].sum()
    area_broken_no_small = df_broken.loc[
        df_broken["area"] > min_threshold, "area"
    ].sum()
    ret["area_broken"] = area_broken
    ret["proportion_broken"] = area_broken / area_total
    ret[f"area_broken>{min_threshold}"] = area_broken_no_small
    ret[f"proportion_broken>{min_threshold}"] = area_broken_no_small / area_total
    return ret


def get_features(
    ccj_img,
    seg_img,
    broken_img,
    blobs_data,
    skeleton_data,
    degrees_img,
    bt_medial_axis_img,
    bt_voronoi_img,
    tx_std_img,
    tx_entropy_img,
):
    """Utilitary function that gets all the features: area, skeleton, nodes
    degrees, branch thickness and texture features.

    Args:
        ccj_img (np.ndarray): Original cell cell junction image.
        seg_img (np.ndarray): Segmented cell cell junction image.
        broken_img (np.ndarray): Broken image.
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
    img_area = ccj_img.shape[0] * ccj_img.shape[1]

    area_fts = get_area_features(seg_img)
    blobs_fts = get_blobs_features(blobs_data)
    sk_fts = get_skeleton_features(skeleton_data)
    dg_fts = get_nodes_degrees_features(degrees_img)
    gd_fts = get_global_dispersion_features(ccj_img, seg_img)
    nb_fts = get_nuclei_blobs_features(
        blobs_data, total_area=img_area, min_area=2500, max_area=70000
    )
    bk_fts = get_broken_features(broken_img)

    features.update(area_fts)
    features.update(blobs_fts)
    features.update(sk_fts)
    features.update(dg_fts)
    features.update(nb_fts)
    features.update(gd_fts)
    features.update(bk_fts)

    map_names_imgs = {
        "branch_thickness_medial_axis": bt_medial_axis_img,
        "branch_thickness_voronoi": bt_voronoi_img,
        "texture_std_filter": tx_std_img,
        "texture_entropy_filter": tx_entropy_img,
    }

    for ft_name, img in map_names_imgs.items():
        features.update(get_statistics_features(img, ft_name))

    return features
