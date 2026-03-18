import numpy as np
from applyHomography import apply_homography
from scipy.ndimage import map_coordinates


def backward_warp_img(src_img, resultToSrc_H, dest_canvas_width_height):
    src_height = src_img.shape[0]
    src_width = src_img.shape[1]
    src_channels = src_img.shape[2]
    dest_width = dest_canvas_width_height[0]
    dest_height = dest_canvas_width_height[1]

    result_img = np.zeros((dest_height, dest_width, src_channels))
    mask = np.zeros((dest_height, dest_width), dtype=bool)

    # this is the overall region covered by result_img
    dest_X, dest_Y = np.meshgrid(np.arange(1, dest_width + 1),
                                 np.arange(1, dest_height + 1))

    # map result_img region to src_img coordinate system using the given homography
    src_pts = apply_homography(resultToSrc_H, np.column_stack(
        [dest_X.ravel(), dest_Y.ravel()]))
    src_X = src_pts[:, 0].reshape(dest_height, dest_width)
    src_Y = src_pts[:, 1].reshape(dest_height, dest_width)

    # ---------------------------
    # START ADDING YOUR CODE HERE
    # ---------------------------

    # Set 'mask' to True where the mapped source coordinates are valid
    valid = (src_X >= 1) & (src_X <= src_width) & \
            (src_Y >= 1) & (src_Y <= src_height)
    mask = valid

    # Interpolate src_img at valid source coordinates (0-indexed row, col)
    # map_coordinates uses (row, col) order
    rows = src_Y.ravel() - 1  # convert to 0-indexed
    cols = src_X.ravel() - 1

    for c in range(src_channels):
        channel = map_coordinates(src_img[:, :, c], [rows, cols],
                                  order=1, mode='constant', cval=0.0)
        result_channel = channel.reshape(dest_height, dest_width)
        result_channel[~valid] = 0.0
        result_img[:, :, c] = result_channel

    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    return mask, result_img
