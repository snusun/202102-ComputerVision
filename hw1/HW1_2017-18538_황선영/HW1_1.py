import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import warnings

def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """
    # Your code
    pad_height = int(size[0]/2)
    pad_width = int(size[1]/2)

    s = input_image.shape
    padded_shape = (s[0]+size[0]-1, s[1]+size[1]-1, s[2])
    img_pad = np.zeros(padded_shape)

    img_pad[pad_height:s[0]+pad_height, pad_width:s[1]+pad_width,:] = image.copy()
    
    img_pad[:pad_height, pad_width:-pad_width, :] = image[pad_height:0:-1, :, :].copy()
    img_pad[-pad_height:, pad_width:-pad_width, :] = image[-2:-pad_height-2:-1, :, :].copy()
    img_pad[pad_height:s[0]+pad_height, :pad_width, :] = image[:, pad_width:0:-1, :].copy()
    img_pad[pad_height:s[0]+pad_height, -pad_width:, :] = image[:, -2:-pad_width-2:-1, :].copy()
    
    img_pad[:pad_height, :pad_width, :] = image[pad_height:0:-1, pad_width:0:-1, :].copy()
    img_pad[:pad_height, -pad_width:, :] = image[pad_height:0:-1, -2:-pad_width-2:-1, :].copy()
    img_pad[-pad_height:, :pad_width, :] = image[-2:-pad_height-2:-1, pad_width:0:-1, :].copy()
    img_pad[-pad_height:, -pad_width:, :] = image[-2:-pad_height-2:-1, -2:-pad_width-2:-1, :].copy()

    return img_pad


def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """

    # Your code
    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image
    flipped = Kernel[::-1,::-1].copy()
    img_pad = reflect_padding(input_image, Kernel.shape)
    convolved = np.zeros(input_image.shape)

    for y in range(img_pad.shape[1]):
        if y > img_pad.shape[1] - flipped.shape[1]:
            break
        for x in range(img_pad.shape[0]):
            if x > img_pad.shape[0] - flipped.shape[0]:
                break
            for z in range(img_pad.shape[2]):
                convolved[x, y, z] = (flipped * img_pad[x:x + flipped.shape[0], y: y + flipped.shape[1], z]).sum()
            
    return convolved


def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")
    # Your code
    img_pad = reflect_padding(input_image, size)
    median = np.zeros(input_image.shape)
    for y in range(img_pad.shape[1]):
        if y > img_pad.shape[1] - size[1]:
            break
        for x in range(img_pad.shape[0]):
            if x > img_pad.shape[0] - size[0]:
                break
            for z in range(img_pad.shape[2]):
                median[x, y, z] = np.median(img_pad[x:x + size[0], y: y + size[1], z], axis=(0,1))
                z = 0
    return median


def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X direction
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """
    # Your code
    k_height = int(size[0]/2)
    k_width = int(size[1]/2)
    p, q = np.ogrid[-k_height:k_height + 1, -k_width:k_width + 1]
    kernel = np.exp(-(q**2 + p**2))/(2*sigmax*sigmay)
    kernel /= kernel.sum()
    gaussian = convolve(input_image, kernel)
    return gaussian.astype(np.uint8)


if __name__ == '__main__':
    image = np.asarray(Image.open(os.path.join('images', 'baboon.jpeg')).convert('RGB'))
    #image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    #image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_1')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((5,5)) / 25.
    sigmax, sigmay = 5, 5
    
    ret = reflect_padding(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'reflect.jpeg'))
        plt.show()
    
    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'convolve.jpeg'))
        plt.show()
    
    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'median.jpeg'))
        plt.show()
    
    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian.jpeg'))
        plt.show()
        


