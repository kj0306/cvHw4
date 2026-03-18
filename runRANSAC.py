import numpy as np
from computeHomography import compute_homography
from applyHomography import apply_homography


def run_ransac(Xs, Xd, ransac_n, eps):
    num_pts = Xs.shape[0]
    pts_id = np.arange(num_pts)
    inliers_id = np.array([], dtype=int)
    H = np.eye(3)  # H placeholder

    for iter in range(ransac_n):
        # ---------------------------
        # START ADDING YOUR CODE HERE
        # ---------------------------

        # Sample 4 random point correspondences
        sample_idx = np.random.choice(num_pts, 4, replace=False)
        H_candidate = compute_homography(Xs[sample_idx], Xd[sample_idx])

        # Apply homography to all source points
        Xd_pred = apply_homography(H_candidate, Xs)

        # Compute reprojection errors
        dists = np.linalg.norm(Xd_pred - Xd, axis=1)

        # Find inliers
        current_inliers = pts_id[dists < eps]

        if len(current_inliers) > len(inliers_id):
            inliers_id = current_inliers
            H = H_candidate

        # ---------------------------
        # END ADDING YOUR CODE HERE
        # ---------------------------

    # Recompute H with all inliers for a better estimate
    if len(inliers_id) >= 4:
        H = compute_homography(Xs[inliers_id], Xd[inliers_id])

    return inliers_id, H
