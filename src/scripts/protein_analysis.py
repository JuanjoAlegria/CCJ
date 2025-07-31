import sys
import argparse
import pickle
from datetime import datetime
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_projections(
    X,
    y,
    model,
    x_coord,
    y_coord,
    min_x=None,
    max_x=None,
    min_y=None,
    max_y=None,
    n_points=500,
    add_text=False,
    plot_margin=False,
):
    colors = {"NS": "tab:red", "NO": "tab:blue"}
    padding = 1

    if min_x is None:
        min_x = X[x_coord].min() - padding
    if max_x is None:
        max_x = X[x_coord].max() + padding
    if min_y is None:
        min_y = X[y_coord].min() - padding
    if max_y is None:
        max_y = X[y_coord].max() + padding

    xx_mesh, yy_mesh = np.meshgrid(
        np.linspace(min_x, max_x, n_points), np.linspace(min_y, max_y, n_points)
    )

    predictions = model.decision_function(
        np.c_[xx_mesh.ravel(), yy_mesh.ravel()]
    ).reshape(xx_mesh.shape)

    fig, axes = plt.subplots(1, figsize=(15, 10))

    axes.contourf(
        xx_mesh, yy_mesh, np.sign(predictions), cmap=plt.cm.coolwarm, alpha=0.5
    )

    for label in ["NO", "NS"]:
        current_X = X[y == label]
        axes.scatter(
            current_X[x_coord], current_X[y_coord], label=label, color=colors[label]
        )

    if add_text:
        for index, row in X.iterrows():
            x0, y0 = row[x_coord], row[y_coord]
            if (min_x <= x0 <= max_x) and (min_y <= y0 <= max_y):
                axes.text(x0 + padding / 100, y0 + padding / 100, index)
            else:
                print(f"{index} protein exceeds the limits (x={x0}, y={y0})")

    if plot_margin:
        w = model.coef_[0]
        a = -w[0] / w[1]
        xx_lin = np.linspace(min_x, max_x)
        yy_lin = a * xx_lin - (model.intercept_[0]) / w[1]
        margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
        yy_down = yy_lin - np.sqrt(1 + a ** 2) * margin
        yy_up = yy_lin + np.sqrt(1 + a ** 2) * margin

        axes.plot(xx_lin, yy_down, "k--")
        axes.plot(xx_lin, yy_up, "k--")

    axes.set_xlabel(x_coord)
    axes.set_ylabel(y_coord)
    axes.set_xlim((min_x, max_x))
    axes.set_ylim((min_y, max_y))
    axes.legend()
    axes.set_title("Projection of proteins into 2D feature space")
    return fig, axes


def compute_distances(proteins_df, svm, x_col, y_col):
    a, b = svm.coef_[0]
    c = svm.intercept_[0]

    distances = abs(a * proteins_df[x_col] + b * proteins_df[y_col] + c) / np.sqrt(
        a ** 2 + b ** 2
    )
    df_ret = proteins_df.copy()
    df_ret["distance_classifier"] = distances
    return df_ret


def main(df_features_path, model_path, proteins, save_dir, min_x, max_x, min_y, max_y):
    x_col = "j2j_distance_median"
    y_col = "branch_thickness_voronoi_median"
    columns = [x_col, y_col]
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
        svm = model["model"]

    df = pd.read_csv(df_features_path, index_col=0)
    if proteins:
        df = df.loc[proteins]

    X = df[columns]
    y = df["stimulus"]
    df_distances = compute_distances(df, svm, x_col, y_col)
    df_distances = df_distances[
        ["protein", "stimulus", "plate", x_col, y_col, "distance_classifier"]
    ]
    print(df_distances)
    fig, ax = plot_projections(
        X,
        y,
        svm,
        x_col,
        y_col,
        add_text=True,
        plot_margin=True,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )

    plt.show()

    now = datetime.now()

    df_save_path = save_dir.joinpath(f"{now:%Y-%m-%d_%H-%M-%S}_results.csv")
    fig_save_path = save_dir.joinpath(f"{now:%Y-%m-%d_%H-%M-%S}_projection.png")

    df_distances.to_csv(df_save_path)
    fig.savefig(fig_save_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--df_features_path",
        help="Path to the CSV file with the computed features.",
        type=Path,
        required=True,
    )
    PARSER.add_argument(
        "--model_path",
        help="Path to the saved SVM model.",
        type=Path,
        default="models/20210408_top_2_features_svm.pkl",
    )
    PARSER.add_argument(
        "--proteins",
        help="""
        List of proteins to be projected into the SVM space. If not given,
        all the proteins in the CSV file will be projected.""",
        nargs="*",
        type=str,
    )
    PARSER.add_argument(
        "--save_dir",
        help="Folder where the results are going to be stored",
        required=True,
        type=Path,
    )
    PARSER.add_argument(
        "--min_x",
        help="Minimum X coordinate in projection's plot",
        default=None,
        type=float,
    )
    PARSER.add_argument(
        "--max_x",
        help="Maximum X coordinate in projection's plot",
        default=None,
        type=float,
    )
    PARSER.add_argument(
        "--min_y",
        help="Minimum Y coordinate in projection's plot",
        default=None,
        type=float,
    )
    PARSER.add_argument(
        "--max_y",
        help="Maximum Y coordinate in projection's plot",
        default=None,
        type=float,
    )

    FLAGS = PARSER.parse_args()
    if not FLAGS.df_features_path.exists():
        sys.exit("CSV file does not exists!")

    if not FLAGS.model_path.exists():
        sys.exit("Model file does not exists!")

    FLAGS.save_dir.mkdir(exist_ok=True, parents=True)

    print(FLAGS.proteins)

    main(
        FLAGS.df_features_path,
        FLAGS.model_path,
        FLAGS.proteins,
        FLAGS.save_dir,
        FLAGS.min_x,
        FLAGS.max_x,
        FLAGS.min_y,
        FLAGS.max_y,
    )
