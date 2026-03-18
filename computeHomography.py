import numpy as np


def compute_homography(src_pts_nx2, dest_pts_nx2):

    n = src_pts_nx2.shape[0]
    A = np.zeros((n * 2, 9))
    for i in range(n):
        x, y = src_pts_nx2[i]
        xp, yp = dest_pts_nx2[i]
        A[2 * i]     = [-x, -y, -1,  0,  0,  0, xp * x, xp * y, xp]
        A[2 * i + 1] = [ 0,  0,  0, -x, -y, -1, yp * x, yp * y, yp]

    # Eigen decomposition of A^T A
    eigvals, eigvecs = np.linalg.eig(A.T @ A)
    idx = np.argmin(eigvals)
    h = eigvecs[:, idx]

    H_3x3 = h.reshape((3, 3))
    return H_3x3
