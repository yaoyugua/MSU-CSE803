import numpy as np
import os
from common import *


## Image Patches ##
def image_patches(image, patch_size=(16,16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N
    # Output- results: a list of images of size M x N

    # TODO: Use slicing to complete the function
    H,W = image.shape
    M,N = patch_size
    output = []
    for i in range(int(H/M)+1):
        for j in range(int(W/N)+1):
            patch = image[i*M:(i+1)*M,j*N:(j+1)*N]
            patch_normalized = (patch - patch.mean()) / patch.var()
            output.append(patch_normalized)
    return output

## Gaussian Filter ##
def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Reminder to implement convolution and not cross-correlation!
    # Input- image: H x W
    #        kernel: h x w
    # Output- convolve: H x W
    H,W = image.shape
    h,w = kernel.shape
    kernel_T = kernel[::-1,::-1]
    output = np.zeros((H,W))
    image = np.pad(image,((int(h/2),int(h/2)),(int(w/2),int(w/2))))
    for i in range(H):
        for j in range(W):
            output[i,j] = np.dot(image[i:i+h,j:j+w].reshape(-1), kernel_T.reshape(-1))
    return output


## Edge Detection ##
def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    # TODO: Fix kx, ky
    kx = np.array([[1/2,0,-1/2]])  # 1 x 3
    ky = np.array([[1/2],[0],[-1/2]])  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    return grad_magnitude, Ix, Iy


## Sobel Operator ##
def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    # TODO: Use convolve() to complete the function
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Gx = convolve(image,Kx)
    Gy = convolve(image,Ky)
    grad_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=[0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W
    # You are encouraged not to use sobel_operator() in this function.

    # TODO: Use convolve() to complete the function
    output = []
    for alpha in angles:
        kernel = np.array([[np.cos(alpha)+np.sin(alpha),2*np.sin(alpha),-np.cos(alpha)+np.sin(alpha)],\
                [2*np.cos(alpha),0,-2*np.cos(alpha)],\
                [np.cos(alpha)-np.sin(alpha),-2*np.sin(alpha),-np.cos(alpha)-np.sin(alpha)]])
        output.append(convolve(image, kernel))
    return output




def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Image Patches #####
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # Q1
    patches = image_patches(img)
    # print(patches[0].shape)
    chosen_patches = patches[:3]
    # TODO choose a few patches and save them

    save_img(chosen_patches[0], "./image_patches/q1_patch0.png")
    save_img(chosen_patches[1], "./image_patches/q1_patch1.png")
    save_img(chosen_patches[2], "./image_patches/q1_patch2.png")

    # Q2: No code

    ##### Gaussian Filter #####
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # Q1: No code

    # Q2

    # TODO: Calculate the kernel described in the question.  There is tolerance for the kernel.
    sigma2 = 1/(2*np.log(2))
    kernel_gaussian = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            kernel_gaussian[i,j] = 1/(2*np.pi*sigma2)*np.exp(-((i-1)**2+(j-1)**2)/(2*sigma2))


    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # Q3
    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    ########################

    ##### Sobel Operator #####
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Q1: No code

    # Q2
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    # Q3
    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    ########################

    #####LoG Filter#####
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # Q1
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # Q2: No code

    print("LoG Filter is done. ")
    ########################


if __name__ == "__main__":
    main()
