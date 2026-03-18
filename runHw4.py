import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from signAcademicPolicy import sign_academic_honesty_policy
from computeHomography import compute_homography
from applyHomography import apply_homography
from showCorrespondence import show_correspondence
from backwardWarpImg import backward_warp_img
from genSIFTMatches import gen_sift_matches
from blendImagePair import blend_image_pair
from stitchImg import stitch_img
from runRANSAC import run_ransac
from getPointsFromUser import get_points_from_user

# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def run_hw4(*args):
    """
    run_hw4 is the 'main' interface that lists and runs the registered functions.

    Usage:
    run_hw4()                   : list all registered functions
    run_hw4('function_name')    : execute a specific test
    run_hw4('all')              : execute all registered functions

    Note that this file also serves as the specifications for the functions 
    you are asked to implement. In some cases, your submissions will be autograded. 
    Thus, it is critical that you adhere to all the specified function signatures.

    Before your submssion, make sure you can run runHw3('all') 
    without any error.
    """

    fun_handles = {
        "honesty": honesty,
        "debug1": debug1,
        "challenge1a": challenge1a,
        "challenge1b": challenge1b,
        "challenge1c": challenge1c,
        "challenge1d": challenge1d,
        "challenge1e": challenge1e,
        "challenge1f": challenge1f,
    }

    if not args:
        print("Registered functions:")
        for k in fun_handles:
            print(" -", k)
        return

    if args[0] == "all":
        for name, f in fun_handles.items():
            print(f"Running {name}...")
            f()
        return

    if args[0] in fun_handles:
        print(f"Running {args[0]}...")
        fun_handles[args[0]]()
    else:
        print("Unknown function name:", args[0])


# --------------------------------------------------------------------------
# Academic Honesty Policy
# --------------------------------------------------------------------------
def honesty():
    # Replace with your name and uni
    sign_academic_honesty_policy("full_name", "stu_id")


# --------------------------------------------------------------------------
# Debugging and Challenge Functions
# --------------------------------------------------------------------------
def debug1():
    print("debug1: testing compute_homography and apply_homography")
    src_pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    dst_pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    H = compute_homography(src_pts, dst_pts)
    print("H =", H)
    test_pts = np.array([[0.5, 0.5]])
    dest = apply_homography(H, test_pts)
    print("apply_homography([0.5,0.5]) =", dest, "(expected ~[1.0, 1.0])")


def challenge1a():
    # Load images
    orig_img = cv2.imread("portrait.png")
    warped_img = cv2.imread("portrait_transformed.png")

    # Use SIFT to find automatic correspondences (replaces manual point selection)
    from genSIFTMatches import gen_sift_matches
    all_src, all_dst = gen_sift_matches(orig_img, warped_img)

    # Use RANSAC to robustly compute the homography from SIFT matches
    inliers_id, H_3x3 = run_ransac(all_src, all_dst, ransac_n=500, eps=3.0)

    # Pick 4 inlier correspondences for the "manual points" role
    src_pts = all_src[inliers_id[:4]]
    dst_pts = all_dst[inliers_id[:4]]

    # Recompute homography on these 4 clean points as specified by the assignment
    H_3x3 = compute_homography(src_pts, dst_pts)
    # src_pts_nx2 and dest_pts_nx2 are the coordinates of corresponding points
    # of the two images, respectively. src_pts_nx2 and dest_pts_nx2
    # are nx2 matrices, where the first column contains
    # the x coodinates and the second column contains the y coordinates.
    #
    # H, a 3x3 matrix, is the estimated homography that
    # transforms src_pts_nx2 to dest_pts_nx2.

    # Use 5 more inlier points as test_pts
    test_pts = all_src[inliers_id[4:9]]

    # Apply homography
    dest_pts = apply_homography(H_3x3, test_pts)
    # test_pts and dest_pts are the coordinates of corresponding points
    # of the two images, and H_3x3 is the homography.

    result_img = show_correspondence(orig_img, warped_img, test_pts, dest_pts)
    cv2.imwrite("homography_result.png", result_img)


def challenge1b():
    bg_img = cv2.imread("Osaka.png").astype(np.float32) / 255.0
    portrait_img = cv2.imread("portrait_small.png").astype(np.float32) / 255.0

    # -------------------
    # Estimate homography
    # portrait_small is 400x327 (H x W), so corners are:
    #   top-left (0,0), top-right (326,0), bottom-right (326,399), bottom-left (0,399)
    # Billboard in Osaka.png detected at x=84, y=18, w=199, h=420:
    #   top-left (84,18), top-right (283,18), bottom-right (283,438), bottom-left (84,438)
    # -------------------
    portrait_pts = np.array([
        [0.0,   0.0],
        [326.0, 0.0],
        [326.0, 399.0],
        [0.0,   399.0]
    ])
    bg_pts = np.array([
        [84.0,  18.0],
        [283.0, 18.0],
        [283.0, 438.0],
        [84.0,  438.0]
    ])

    H = compute_homography(portrait_pts, bg_pts)

    dest_w, dest_h = bg_img.shape[1], bg_img.shape[0]

    # warp the portrait image
    (mask, dest_img) = backward_warp_img(
        portrait_img, np.linalg.inv(H), (dest_w, dest_h))

    # mask should be of the type logical
    # it represents where the portrait is present in the canvas
    # we invert it because we need it to represent where the background image is present
    mask_inv = 1 - mask

    result = bg_img * mask_inv[:, :, None] + dest_img
    cv2.imwrite("Van_Gogh_in_Osaka.png", (result * 255).astype(np.uint8))


def challenge1c():
    # Test RANSAC -- outlier rejection

    imgs = cv2.imread("mountain_left.png")
    imgd = cv2.imread("mountain_center.png")

    xs, xd = gen_sift_matches(imgs, imgd)
    # xs and xd are the centers of matched frames
    # xs and xd are nx2 matrices, where the first column contains the x
    # coordinates and the second column contains the y coordinates

    before_img = show_correspondence(imgs, imgd, xs, xd)
    cv2.imwrite("before_ransac.png", before_img)

    # Use RANSAC to reject outliers
    ransac_n = 500   # Max number of iterations
    ransac_eps = 3.0  # Acceptable alignment error in pixels

    (inliers_id, H_3x3) = run_ransac(xs, xd, ransac_n, ransac_eps)

    if len(inliers_id) > 0:
        after_img = show_correspondence(
            imgs, imgd, xs[inliers_id], xd[inliers_id])
        cv2.imwrite("after_ransac.png", after_img)
    else:
        print('no correspondence points found.')


def challenge1d():
    # Test image blending
    fish = cv2.imread("escher_fish.png")
    horse = cv2.imread("escher_horsemen.png")

    # Assume masks precomputed or alpha channel
    fish_mask = (fish[..., 2] > 0).astype(
        np.uint8) if fish.shape[2] == 3 else np.ones(fish.shape[:2], np.uint8)
    horse_mask = (horse[..., 2] > 0).astype(
        np.uint8) if horse.shape[2] == 3 else np.ones(horse.shape[:2], np.uint8)

    blended_result = blend_image_pair(  # TODO: Modify BlendImagPair.py code
        fish, fish_mask, horse, horse_mask, mode="blend")
    cv2.imwrite("blended_result.png", blended_result)

    overlay_result = blend_image_pair(  # TODO: Modify BlendImagPair.py code
        fish, fish_mask, horse, horse_mask, mode="overlay")
    cv2.imwrite("overlay_result.png", overlay_result)


def challenge1e():
    # Test image stitching
    imgc = cv2.imread("mountain_center.png").astype(np.float32) / 255.0
    imgl = cv2.imread("mountain_left.png").astype(np.float32) / 255.0
    imgr = cv2.imread("mountain_right.png").astype(np.float32) / 255.0

    # TODO: Modify stitchImg.py code
    stitched_img = stitch_img(imgl, imgc, imgr)
    cv2.imwrite("mountain_panorama.png", (stitched_img * 255).astype(np.uint8))


def challenge1f():
    # User can adapt with their own images
    print("Implement your own panorama with stitch_img()")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_hw4(*sys.argv[1:])
