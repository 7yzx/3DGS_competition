import cv2
import numpy as np



def get_rectified_stereo_data(img0, img1, intr0, intr1, extr0, extr1, mask0=None, mask1=None):
    H, W = img0.shape[:2]
    r0, t0 = extr0[:3, :3], extr0[:3, 3:]
    r1, t1 = extr1[:3, :3], extr1[:3, 3:]
    inv_r0 = r0.T
    inv_t0 = -r0.T @ t0
    E0 = np.eye(4)
    E0 = np.eye(4)
    E0[:3, :3], E0[:3, 3:] = inv_r0, inv_t0
    E1 = np.eye(4)
    E1[:3, :3], E1[:3, 3:] = r1, t1
    E = E1 @ E0
    R, T = E[:3, :3], E[:3, 3]
    dist0, dist1 = np.zeros(4), np.zeros(4)
    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
        intr0, dist0, intr1, dist1, (W, H), R, T, flags=0)
    new_extr0 = R0 @ extr0
    new_intr0 = P0[:3, :3]
    new_extr1 = R1 @ extr1
    new_intr1 = P1[:3, :3]
    Tf_x = np.array(P1[0, 3])

    rectify_mat0_x, rectify_mat0_y = cv2.initUndistortRectifyMap(
        intr0, dist0, R0, P0, (W, H), cv2.CV_32FC1)
    new_img0 = cv2.remap(img0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    if mask0 is not None:
        new_mask0 = cv2.remap(mask0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)

    rectify_mat1_x, rectify_mat1_y = cv2.initUndistortRectifyMap(
        intr1, dist1, R1, P1, (W, H), cv2.CV_32FC1)
    new_img1 = cv2.remap(img1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
    if mask1 is not None:
        new_mask1 = cv2.remap(mask1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

    stereo_data = {
        'img0': new_img0,
        'img1': new_img1,
        'mask0': new_mask0 if mask0 is not None else None,
        'mask1': new_mask1 if mask1 is not None else None,
        'intr0': new_intr0,
        'intr1': new_intr1,
        'extr0': new_extr0,
        'extr1': new_extr1,
        'baseline': np.abs(Tf_x / P0[0, 0]),
    }

    return stereo_data