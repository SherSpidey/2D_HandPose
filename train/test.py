import numpy as np

def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)

def make_heatmaps_from_joints(input_size, heatmap_size, gaussian_variance, batch_joints):
    # Generate ground-truth heatmaps from ground-truth 2d joints
    scale_factor = input_size // heatmap_size
    batch_gt_heatmap_np = []
    for i in range(batch_joints.shape[0]):
        gt_heatmap_np = []d.
        invert_heatmap_np = np.ones(shape=(heatmap_size, heatmap_size))
        for j in range(batch_joints.shape[1]):
            cur_joint_heatmap = make_gaussian(heatmap_size,
                                              gaussian_variance,
                                              center=(batch_joints[i][j] // scale_factor))
            gt_heatmap_np.append(cur_joint_heatmap)
            invert_heatmap_np -= cur_joint_heatmap
        gt_heatmap_np.append(invert_heatmap_np)
        batch_gt_heatmap_np.append(gt_heatmap_np)
    batch_gt_heatmap_np = np.asarray(batch_gt_heatmap_np)
    batch_gt_heatmap_np = np.transpose(batch_gt_heatmap_np, (0, 2, 3, 1))

    return batch_gt_heatmap_np

