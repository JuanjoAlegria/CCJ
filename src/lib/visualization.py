import io
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def draw_centroids(centroids_df, img, write_area=True,
                   cent_color='red', cent_radius=5,
                   font_color='lightblue', font_size=10):

    _, axes = get_plot_with_img(img)
    for _, row in centroids_df.iterrows():
        axes.add_artist(plt.Circle(
            (row['x'], row['y']), radius=cent_radius, color=cent_color))
        if write_area:
            axes.text(row['x'], row['y'], row['m00'],
                    fontsize=font_size, color=font_color)
    return save_ax_nosave(axes)


def draw_skeleton_lines(skeleton_df, img, color, thickness=5):
    copy = img.copy()
    for _, row in skeleton_df.iterrows():
        x0 = int(row['image-coord-src-0'])
        y0 = int(row['image-coord-src-1'])
        x1 = int(row['image-coord-dst-0'])
        y1 = int(row['image-coord-dst-1'])
        # Note: the points are inverted!
        cv2.line(copy, (y0, x0), (y1, x1), color=color, thickness=thickness)
    return copy


def draw_nodes(img, degrees, cmap, norm):
    copy = img.copy()
    nodes_coords = np.where(degrees > 2)

    for x, y in zip(*nodes_coords):
        color = (255 * np.array(cmap(norm(degrees[x,y]))))[:-1]
        copy = cv2.circle(copy, (y, x), radius=10, color=color, thickness=-1)

    return copy

def get_plot_with_img(img):
    dpi = mpl.rcParams['figure.dpi']
    height, width = img.shape[0], img.shape[1]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(img, cmap='gray')

    return fig, ax

def save_ax_nosave(ax, **kwargs):
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    buff = io.BytesIO()
    plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox,  **kwargs)
    ax.axis("on")
    buff.seek(0)
    im = plt.imread(buff)
    return im
