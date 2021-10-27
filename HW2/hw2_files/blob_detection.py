from common import * 
import matplotlib.pyplot as plt
import numpy as np 
from filters import *

def gaussian_filter(image, sigma):
    # Given an image, apply a Gaussian filter with the input kernel size
    # and standard deviation 
    # Input-    image: image of size HxW
    #           sigma: scalar standard deviation of Gaussian Kernel
    # Output-   Gaussian filtered image of size HxW
    H, W = image.shape 
    # -- good heuristic way of setting kernel size 
    kernel_size = int(2 * np.ceil(2*sigma) + 1) 

    # make sure that kernel size isn't too big and is odd 
    kernel_size = min(kernel_size, min(H,W)//2) 
    if kernel_size % 2 == 0: kernel_size = kernel_size + 1

    #TODO implement gaussian filtering with size kernel_size x kernel_size 
    # feel free to use your implemented convolution function or a convolution function from a library 
    kernel_gaussian = np.zeros([kernel_size, kernel_size])
    center = (kernel_size - 1) / 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel_gaussian[i,j] = 1/(2*np.pi*sigma**2)*np.exp(-((i-center)**2+(j-center)**2)/(2*sigma**2))   
    image_filtered = convolve(image, kernel_gaussian) 
    return image_filtered

def scale_space(image, min_sigma, k=np.sqrt(2), S=8):
    # Calcualtes a DoG scale space of the image
    # Input-    image: image of size HxW
    #           min_sigma: smallest sigma in scale space
    #           k: scalar multiplier for scale space
    #           S: number of scales considers
    # Output-   Scale Space of size HxWx(S-1)
    H,W = image.shape
    output = np.zeros([H,W,S-1])
    for i in range(S-1):
        sigma_small = np.power(k,i)
        sigma_large = np.power(k,i+1)
        gauss_1 = gaussian_filter(image,sigma_small)
        gauss_2 = gaussian_filter(image,sigma_large)
        # calculate difference of gaussians
        output[:,:,i] = gauss_1 - gauss_2
    return output


##### You shouldn't need to edit the following 3 functions 
def find_maxima(scale_space, k_xy=5, k_s=1):
    # Extract the peak x,y locations from scale space
    # Input-    scale_space: Scale space of size HxWxS
    #           k: neighborhood in x and y
    #           ks: neighborhood in scale
    # Output-   list of (x,y) tuples; x<W and y<H
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None] 

    H,W,S = scale_space.shape
    maxima = []
    for i in range(H):
        for j in range(W):
            for s in range(S):
                # extracts a local neighborhood of max size (2k_xy+1, 2k_xy+1, 2k_s+1)
                neighbors = scale_space[max(0, i-k_xy):min(i+k_xy,H), 
                                        max(0, j-k_xy):min(j+k_xy,W), 
                                        max(0, s-k_s) :min(s+k_s,S)]
                mid_pixel = scale_space[i,j,s]
                num_neighbors = np.prod(neighbors.shape) - 1
                # if mid_pixel is larger than all the neighbors; append maxima 
                if np.sum(mid_pixel > neighbors) == num_neighbors:
                    maxima.append( (i,j,s) )
    return maxima

def visualize_scale_space(scale_space, min_sigma, k, file_path=None):
    # Visualizes the scale space
    # Input-    scale_space: scale space of size HxWxS
    #           min_sigma: the minimum sigma used 
    #           k: the sigma multiplier 
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None] 
    H, W, S = scale_space.shape

    # number of subplots
    p_h = int(np.floor(np.sqrt(S))) 
    p_w = int(np.ceil(S/p_h))
    for i in range(S):
        plt.subplot(p_h, p_w, i+1)
        plt.axis('off')
        plt.title('{:.1f}:{:.1f}'.format(min_sigma * k**i, min_sigma * k**(i+1)))
        plt.imshow(scale_space[:, :, i])

    # plot or save to fig 
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()    

def visualize_maxima(image, maxima, min_sigma, k, file_path=None):
    # Visualizes the maxima on a given image
    # Input-    image: image of size HxW
    #           maxima: list of (x,y) tuples; x<W, y<H
    #           file_path: path to save image. if None, display to screen
    # Output-   None 
    H, W = image.shape
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for maximum in maxima:
        y,x,s= maximum 
        assert x < W and y < H and x >= 0 and y >= 0
        radius = np.sqrt(2 * min_sigma * (k ** s))
        circ = plt.Circle((x, y), radius, color='r', fill=False)
        ax.add_patch(circ)

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()    


def main():
    """
    image = read_img('polka.png')

    ### -- Detecting Polka Dots -- ## 
    print("Detect small polka dots")
    # -- Detect Small Circles
    sigma_1, sigma_2 = 2, 4
    gauss_1 = gaussian_filter(image,sigma_1)
    gauss_2 = gaussian_filter(image,sigma_2)

    # calculate difference of gaussians
    DoG_small = gauss_1 - gauss_2

    # visualize maxima 
    maxima = find_maxima(DoG_small, k_xy=int(sigma_1))
    visualize_scale_space(DoG_small, sigma_1, sigma_2/sigma_1,'polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2/sigma_1, 'polka_small.png')
    
    # -- Detect Large Circles
    print("Detect large polka dots")
    sigma_1, sigma_2 = 10, 20
    gauss_1 = gaussian_filter(image,sigma_1)
    gauss_2 = gaussian_filter(image,sigma_2)

    # calculate difference of gaussians 
    DoG_large = gauss_1 - gauss_2
    
    # visualize maxima 
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2/sigma_1, 'polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2/sigma_1, 'polka_large.png')


    ## -- TODO Implement scale_space() and try to find both polka dots 

    DoG_scale = scale_space(image, 2)
    visualize_scale_space(DoG_scale, 2, np.sqrt(2), 'polka_scale_DoG.png')
    for k_xy_value in [1,5,9,15]:
        for k_s in [1,2,3]:
            maxima = find_maxima(DoG_scale, k_xy = k_xy_value, k_s=k_s)
            visualize_maxima(image, maxima, 2, np.sqrt(2), 'polka_scale_kxy{}_ks{}.png'.format(k_xy_value,k_s))
    """

    ## -- TODO Try to find the cells in any of the cell images in vgg_cells 
    for _id in [1,2,4,5]:
        id_img = _id
        image = read_img('cells/00{}cell.png'.format(id_img))
        image = (image>(np.max(image)+np.min(image))/2)*255
        print(np.max(image),np.min(image))
        min_sigma = 2
        k = np.sqrt(2)
        DoG_scale = scale_space(image, min_sigma)
        maxima = find_maxima(DoG_scale, k_xy = 8, k_s = 2)
        print(len(maxima))
        visualize_maxima(image, maxima, min_sigma, k, '00{}cell_maxima.png'.format(id_img))

if __name__ == '__main__':
    main()
