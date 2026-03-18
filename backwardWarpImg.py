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

    # Set 'mask' to the correct values based on src_pts.

    # fill the right region in 'result_img' with the src_img

    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    return mask, result_img
