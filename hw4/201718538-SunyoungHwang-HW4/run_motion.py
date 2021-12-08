import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.
    #dp = 0 # you should delete this

    # warp image2
    warp_mg = np.meshgrid(range(img1.shape[1]), range(img1.shape[0]))
    warp_x = p[0] * warp_mg[0] + p[2] * warp_mg[1] + p[4] * np.ones(warp_mg[0].shape)
    warp_y = p[1] * warp_mg[0] + p[3] * warp_mg[1] + p[5] * np.ones(warp_mg[0].shape)
    I_w = RectBivariateSpline(np.arange(img2.shape[0]), np.arange(img2.shape[1]), img2).ev(warp_y, warp_x)

    # create gradient vector
    grad = np.hstack((Gx.reshape(-1, 1), Gy.reshape(-1, 1)))
    grad /= max(Gx.max(), Gy.max())
 
    # compute jacobian matrix
    dt = np.zeros((img1.shape[0]*img1.shape[1], 6))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            jac = np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])
            dt[img1.shape[1] * i + j] = np.matmul(grad[img1.shape[1] * i + j], jac)
    dt = dt * max(Gx.max(), Gy.max())

    # compute hessian matrix
    H = np.matmul(dt.T, dt)

    # compute delta p
    dp = np.matmul(np.matmul(np.linalg.inv(H), dt.T), (img1 - I_w).flatten()).flatten()

    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5) # do not modify this
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5) # do not modify this
    
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.
    #moving_image = np.abs(img2 - img1) # you should delete this
    
    th_hi = 0.2 * 256 # you can modify this
    th_lo = 0.15 * 256 # you can modify this

    ##### code start 
    img2_ls_x = np.linspace(0, img2.shape[1]-1, img2.shape[1])
    img2_ls_y = np.linspace(0, img2.shape[0]-1, img2.shape[0])
    img2_mg = np.meshgrid(img2_ls_x, img2_ls_y)

    # compute p
    p = np.array([1, 0, 0, 1, 0, 0])
    dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
    p = p + dp

    # warp img2
    warp_x = p[0] * img2_mg[0] + p[2] * img2_mg[1] + p[4] * np.ones(img2_mg[0].shape)
    warp_y = p[1] * img2_mg[0] + p[3] * img2_mg[1] + p[5] * np.ones(img2_mg[0].shape)
    warp = RectBivariateSpline(img2_ls_y, img2_ls_x, img2).ev(warp_y, warp_x)
    moving_image = np.abs(img1 - warp)

    ### END CODE HERE ###
    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

if __name__ == "__main__":
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
    