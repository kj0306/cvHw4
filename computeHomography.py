import numpy as np


def compute_homography(src_pts_nx2, dest_pts_nx2):
    
    # TODO: Implement this function to compute the homography matrix H
    # that transforms src_pts_nx2 to dest_pts_nx2.

    # n = src_pts_nx2.shape[0]
    # A = np.zeros((n * 2, 9))
    # for i in range(n):
        

    # Eigen decomposition of A^T A
    eigvals, eigvecs = np.linalg.eig(A.T @ A)
    idx = np.argmin(eigvals)
    h = eigvecs[:, idx]

    H_3x3 = h.reshape((3, 3))
    return H_3x3
