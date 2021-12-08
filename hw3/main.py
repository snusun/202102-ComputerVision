import math
import numpy as np
from PIL import Image

def compute_h(p1, p2):
    # TODO ...
    P = np.zeros((2*p1.shape[0], 9))

    for p in zip(range(p1.shape[0]), p1, p2):
        P[2*p[0]:2*p[0]+2, :] = np.array([[-p[2][0], -p[2][1], -1, 0, 0, 0, p[1][0]*p[2][0], p[1][0]*p[2][1], p[1][0]],
                                [0, 0, 0, -p[2][0], -p[2][1], -1, p[1][1]*p[2][0], p[1][1]*p[2][1], p[1][1]]])

    _, _, Vt = np.linalg.svd(P)
    H = Vt[-1].reshape((3,3))
    
    return H

def compute_h_norm(p1, p2):
    # TODO ...
    n = np.zeros(p1.shape)
    n[:, 0] = 1600
    n[:, 1] = 1200
    p1 = p1 / n
    p2 = p2 / n
    return compute_h(p1, p2)

def warp_image(igs_in, igs_ref, H):
    # TODO ...
    trans = np.matmul(H, np.array([[0, 0, 1, 1],[0, 1, 0, 1],[1, 1, 1, 1]])) 
    trans = trans/trans[2]
    x_min = int(np.min(trans[0])*1600)-1
    x_max = int(np.max(trans[0])*1600)+1
    y_min = int(np.min(trans[1])*1200)-1
    y_max = int(np.max(trans[1])*1200)+1
    igs_warp = np.zeros((y_max-y_min+1, x_max-x_min+1, 3))
    merge_row = max(y_max-y_min+1, igs_ref.shape[0]-y_min)
    merge_col = max(x_max-x_min+1, igs_ref.shape[1]-x_min)
    igs_merge = np.zeros((merge_row, merge_col, 3))

    for i in range(x_min, x_max+1):
        for j in range(y_min, y_max+1):
            x_nor = i/1600
            y_nor = j/1200
            A = np.array([[x_nor * H[2][0] - H[0][0], x_nor * H[2][1] - H[0][1]], [y_nor * H[2][0] - H[1][0], y_nor * H[2][1] - H[1][1]]])
            b = np.array([H[0][2] - x_nor * H[2][2], H[1][2] - y_nor * H[2][2]])
            t_y, t_x = np.array([1600, 1200]) * np.linalg.solve(A, b)
            in_i = round(t_y)
            in_j = round(t_x)
            if in_i < 1 or in_i >= 1600 or in_j < 1 or in_j >= 1200:
                igs_warp[j-y_min, i-x_min] = 0
            else:
                igs_warp[j-y_min, i-x_min] = igs_in[in_j, in_i]
    
    igs_merge[0:y_max-y_min+1, 0:x_max-x_min+1] = igs_warp
    for i in range(0, igs_ref.shape[1]): 
        for j in range(0, igs_ref.shape[0]):
            igs_merge[j-y_min, i-x_min] = igs_ref[j,i]

    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # TODO ...
    n = np.zeros(p1.shape)  
    n[:, 0] = 1920
    n[:, 1] = 1056
    p1 = p1 / n
    p2 = p2 / n
    H = compute_h(p2, p1)

    trans = np.matmul(H, np.array([[0, 0, 1, 1],[0, 1, 0, 1],[1, 1, 1, 1]])) 
    trans = trans/trans[2]
    x_min = int(np.min(trans[0])*1920)-1
    x_max = int(np.max(trans[0])*1920)+1
    y_min = int(np.min(trans[1])*1056)-1
    y_max = int(np.max(trans[1])*1056)+1
    igs_rec = np.zeros((y_max-y_min+1, x_max-x_min+1, 3))
    
    for i in range(x_min, x_max+1):
        for j in range(y_min, y_max+1):
            x_nor = i/1920
            y_nor = j/1056
            A = np.array([[x_nor*H[2][0] - H[0][0], x_nor*H[2][1] - H[0][1]], [y_nor*H[2][0] - H[1][0], y_nor*H[2][1] - H[1][1]]])
            b = np.array([H[0][2] - x_nor*H[2][2], H[1][2] - y_nor*H[2, 2]])
            t_y, t_x = np.array([1920, 1056]) * np.linalg.solve(A, b)
            in_i = round(t_y)
            in_j = round(t_x)
            if in_i < 1 or in_i >= 1920 or in_j < 1 or in_j >= 1056:
                igs_rec[j-y_min, i-x_min] = 0 
            else:
                igs_rec[j-y_min, i-x_min] = igs[in_j, in_i]
                
    return igs_rec

def set_cor_mosaic():
    # TODO ...
    p_in = np.array([[1283, 418],
                     [1444, 404],
                     [1283, 511],
                     [1446, 506],
                     [1239, 541],
                     [1296, 543],
                     [1255, 964],
                     [1284, 967],
                     [1334, 919],
                     [1461, 585],
                     [1466, 723]])
    p_ref = np.array([[535, 423],
                     [678, 424],
                     [536, 515],
                     [679, 513],
                     [493, 544],
                     [548, 545],
                     [507, 948],
                     [536, 951],
                     [583, 898],
                     [691, 584],
                     [694, 711]])
    return p_in, p_ref

def set_cor_rec():
    # TODO ...
    c_in = np.array([[1064, 166],
                     [1403, 127],
                     [1047, 869],
                     [1397, 888]])

    c_ref = np.array([[0, 0],
                      [200, 0],
                      [0, 400],
                      [200, 400]])
    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_mergeed.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
