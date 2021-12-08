import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils

def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    g_pyramid = [input_image]
    for i in range(1, level+1):
        if i==1:
            g_pyramid.append(utils.down_sampling(input_image))
        else:
            g_pyramid.append(utils.down_sampling(g_pyramid[i-1]))
    return g_pyramid


def laplacian_pyramid(gaussian_pyramid):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    l_pyramid = []
    for i in range(len(gaussian_pyramid)):
        if i<len(gaussian_pyramid)-1:
            l_pyramid.append(utils.safe_subtract(gaussian_pyramid[i].astype(np.int16), utils.up_sampling(gaussian_pyramid[i+1]).astype(np.int16)))
        else:
            l_pyramid.append(gaussian_pyramid[i])
    return l_pyramid

def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """
    # Your code
    gau1 = gaussian_pyramid(image1, level)
    gau2 = gaussian_pyramid(image2, level)
    gauM = gaussian_pyramid(mask, level)
    lap1 = laplacian_pyramid(gau1)
    lap2 = laplacian_pyramid(gau2)

    mix = []
    for a,b,m in zip(lap1,lap2,gauM):
        B = np.clip((b * m/255), 0, 255).astype('uint8')
        A = np.clip((a * 1-(m/255)), 0, 255).astype('uint8')
        #print(1-gm/256)
        mix.append(utils.safe_add(A, B))
    mix.reverse()

    blend = mix[0]
    for i in range(1,level+1):
        blend = utils.up_sampling(blend)
        blend = utils.safe_add(blend, mix[i])
    return blend.astype('uint8')


if __name__ == '__main__':
    hand = np.asarray(Image.open(os.path.join('images', 'hand.jpeg')).convert('RGB'))
    flame = np.asarray(Image.open(os.path.join('images', 'flame.jpeg')).convert('RGB'))
    mask = np.asarray(Image.open(os.path.join('images', 'mask.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3

    
    plt.figure()
    plt.imshow(Image.open(os.path.join('images', 'direct_concat.jpeg')))
    plt.axis('off')
    plt.savefig(os.path.join(logdir, 'direct.jpeg'))
    plt.show()

    
    ret = gaussian_pyramid(hand, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian_pyramid.jpeg'))
        plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis('off')
            plt.savefig(os.path.join(logdir, 'laplacian_pyramid.jpeg'))
            plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'blended.jpeg'))
        plt.show()
