import skimage
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from skimage.util import invert
from skimage.measure import label, regionprops_table, perimeter
from skimage.morphology import dilation, square


def get_regionprops_df(seg_img, labeled_img):
    seg_img = dilation(seg_img, square(3))
    blobs_at_borders = np.unique(
        np.array(
            [labeled_img[0], labeled_img[-1], labeled_img[:, 0], labeled_img[:, -1]]
        )
    )

    props = regionprops_table(
        labeled_img,
        intensity_image=seg_img,
        properties=(
            "label",
            "centroid",
            "area",
            "perimeter",
            "convex_area",
            "solidity",
            "major_axis_length",
            "minor_axis_length",
            "convex_image",
        ),
    )
    df_props = pd.DataFrame(props)

    # Compute other features: convex_perimeter, compactness, convexity and
    # elongation
    df_props["convex_perimeter"] = df_props["convex_image"].apply(
        lambda img: perimeter(img)
    )
    df_props["compactness"] = (
        4 * np.pi * df_props["area"] / (df_props["perimeter"] ** 2)
    )
    df_props["convexity"] = df_props["convex_perimeter"] / df_props["perimeter"]
    df_props["elongation"] = (
        df_props["major_axis_length"] / df_props["minor_axis_length"]
    )

    df_props["is_at_border"] = df_props["label"].isin(blobs_at_borders)

    # Drop unuseful columns and rename others to conform to the previous API
    df_props = df_props.drop(columns="convex_image").rename(
        columns={"convex_area": "hull_area", "convex_perimeter": "hull_perimeter"}
    )

    # Clean the data, removing artifacts
    index_artifacts = (
        (df_props["hull_area"] <= 1)
        | (df_props["hull_perimeter"] <= 1)
        | (df_props["major_axis_length"] < 1)
        | (df_props["minor_axis_length"] < 1)
    )

    df_props = df_props[~index_artifacts]

    df_errors = df_props[
        df_props.isna().any(axis=1) | df_props.isin([np.inf, -np.inf]).any(axis=1)
    ]

    assert len(df_errors) == 0
    return df_props


def get_blobs_measurements(seg_img, return_img=False):
    """Computes the properties of each blob.

    The blobs at the border of the image are not taken into account.

    Args:
        seg_img (2-D np.array): Binary segmented image

    Returns:
        df_props (pd.DataFrame): Dataframe with the properties of each region
    """
    seg_inv = invert(seg_img)
    labeled_img = label(seg_inv)
    df_props = get_regionprops_df(seg_inv, labeled_img)

    if return_img:
        return df_props, labeled_img
    else:
        return df_props
