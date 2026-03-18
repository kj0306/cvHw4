import numpy as np
from scipy.ndimage import distance_transform_edt as bwdist


def blend_image_pair(wrapped_imgs, masks, wrapped_imgd, maskd, mode):
    Hs, Ws, Cs = wrapped_imgs.shape
    Hd, Wd, Cd = wrapped_imgd.shape

    assert (Hs == Hd) and (Ws == Wd) and (Cs == Cd)

    assert wrapped_imgs.dtype == wrapped_imgd.dtype
    assert wrapped_imgs.dtype in [np.uint8, np.float32, np.float64]

    out_img = np.zeros((Hs, Ws, Cs), dtype=np.float64)
    input_type = wrapped_imgs.dtype

    # convert to float64 to avoid overflow/underflow when multiplying with
    # the weighted mask
    wrapped_imgs = wrapped_imgs.astype(np.float64)
    wrapped_imgd = wrapped_imgd.astype(np.float64)

    binary_mask_s = masks > 0
    binary_mask_d = maskd > 0

    if mode == "blend":
        # Compute distance transforms for weighting
        dist_s = bwdist(binary_mask_s)
        dist_d = bwdist(binary_mask_d)

        total = dist_s + dist_d
        # Avoid division by zero
        total_safe = np.where(total == 0, 1.0, total)

        weight_s = np.zeros((Hs, Ws), dtype=np.float64)
        weight_d = np.zeros((Hs, Ws), dtype=np.float64)

        # In overlap region: weighted blend
        overlap = binary_mask_s & binary_mask_d
        weight_s[overlap] = dist_s[overlap] / total_safe[overlap]
        weight_d[overlap] = dist_d[overlap] / total_safe[overlap]

        # Only in s: full weight to s
        only_s = binary_mask_s & ~binary_mask_d
        weight_s[only_s] = 1.0

        # Only in d: full weight to d
        only_d = binary_mask_d & ~binary_mask_s
        weight_d[only_d] = 1.0

    for c in range(Cs):
        channel_out = np.zeros((Hs, Ws), dtype=np.float64)
        S = wrapped_imgs[:, :, c]
        D = wrapped_imgd[:, :, c]
        if mode == "overlay":
            # s first, then d overwrites s wherever there is overlap.
            channel_out[binary_mask_s] = S[binary_mask_s]
            channel_out[binary_mask_d] = D[binary_mask_d]
        elif mode == "blend":
            # ---------------------------
            # ADD YOUR CODE HERE
            # ---------------------------
            channel_out = S * weight_s + D * weight_d
        out_img[:, :, c] = channel_out

    # convert out_img to right type
    if input_type == np.uint8:
        out_img = np.clip(np.round(out_img), 0, 255).astype(np.uint8)
    elif input_type == np.float32:
        out_img = out_img.astype(np.float32)

    return out_img
