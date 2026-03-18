import numpy as np
import matplotlib.pyplot as plt


def show_correspondence(orig_img, warped_img, src_pts_nx2, dest_pts_nx2):

    n = src_pts_nx2.shape[0]

    Hs, Ws, _ = orig_img.shape
    Hd, Wd, _ = warped_img.shape

    middle_space = 10
    H = max(Hs, Hd)
    W = Ws + middle_space + Wd
    result_img = np.zeros((H, W, orig_img.shape[2]), dtype=np.uint8)

    offset_s = [0, (H - Hs) // 2]
    offset_d = [Ws + middle_space, (H - Hd) // 2]

    result_img[offset_s[1]:offset_s[1]+Hs,
               offset_s[0]:offset_s[0]+Ws, :] = orig_img
    result_img[offset_d[1]:offset_d[1]+Hd,
               offset_d[0]:offset_d[0]+Wd, :] = warped_img

    dpi = 100
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(result_img)

    for i in range(n):
        xs = offset_s[0] + src_pts_nx2[i, 0]
        ys = offset_s[1] + src_pts_nx2[i, 1]
        xd = offset_d[0] + dest_pts_nx2[i, 0]
        yd = offset_d[1] + dest_pts_nx2[i, 1]
        ax.plot([xs, xd], [ys, yd], linewidth=1, color='r')

    plt.show()

    # Convert figure to image array
    fig.canvas.draw()
    # buffer_rgba is in RGB order; convert to BGR so cv2.imwrite saves correctly
    result_img = np.array(fig.canvas.buffer_rgba())[..., 2::-1]
    plt.close(fig)

    return result_img
