import os
from common import read_img, save_img
import matplotlib.pyplot as plt
import numpy as np
from filters import *

def corner_score(image, u=5, v=5, window_size=(5,5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return the function E(u,v) for window size W
    # corner detector score for that pixel.
    # Input- image: H x W
    #        u: a scalar for x offset
    #        v: a scalar for y offset
    #        window_size: a tuple for window size
    #
    # Output- results: a image of size H x W
    # Use zero-padding to handle window values outside of the image.
    H,W = image.shape
    h,w = window_size
    ph = int(h/2)+abs(u)
    pw = int(w/2)+abs(v)
    im = np.pad(image,((ph,ph),(pw,pw)), 'constant', constant_values = (0,0)) # pad the original image with 0.
    output = np.zeros(image.shape)

    for i in range(H):
        for j in range(W):
            w1 = im[i+ph-int(h/2):i+ph+int(h/2), j+pw-int(w/2):j+pw+int(w/2)]
            w2 = im[i+ph-int(h/2)+u:i+ph+int(h/2)+u, j+pw-int(w/2)+v:j+pw+int(w/2)+v]
            output[i,j] = np.sum((w2-w1)**2)
    return output


def harris_detector(image, window_size=(5,5)):
    # Given an input image, calculate the Harris Detector score for all pixels
    # Input- image: H x W
    # Output- results: a image of size H x W
    #
    # You can use same-padding for intensity (or zero-padding for derivatives)
    # to handle window values outside of the image.

    ## compute the derivatives
    kx = np.array([[1/2,0,-1/2]])  # 1 x 3
    ky = np.array([[1/2],[0],[-1/2]])  # 3 x 1
    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    H,W = image.shape
    h,w = window_size
    ph = int(h/2)
    pw = int(w/2)
    im = np.pad(image,((ph,ph),(pw,pw)), 'constant', constant_values = (0,0))

    M = np.zeros([H,W,3])
    window = 1/273 * np.array([[1,4,7,4,1],\
    [4,16,26,16,4],\
    [7,26,41,26,7],\
    [4,16,26,16,4],\
    [1,4,7,4,1]])
    M[:,:,0] = convolve(Ixx, window)
    M[:,:,1] = convolve(Ixy, window)
    M[:,:,2] = convolve(Iyy, window)
    response = M[:,:,0]*M[:,:,2]-M[:,:,1]**2 - 0.05 * (M[:,:,0]+M[:,:,2])**2

    # For each location of the image, construct the structure tensor and calculate the Harris response
    # response = None

    return response

def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Feature Detection #####
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # define offsets and window size and calulcate corner score
    u, v, W = -5, 0, (5,5) # left
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score_left.png")
    u, v, W = 5, 0, (5,5) # right
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score_right.png")
    u, v, W = 0, 5, (5,5) # up
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score_up.png")
    u, v, W = 0, -5, (5,5) # down
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score_down.png")

    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")

if __name__ == "__main__":
    main()
