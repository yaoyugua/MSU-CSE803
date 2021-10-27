import numpy as np
from matplotlib import pyplot as plt
from common import *
from numpy.linalg import lstsq as least_square
import cv2
# feel free to include libraries needed


def homography_transform(X, H):
    # TODO
    # Perform homography transformation on a set of points X
    # using homography matrix H
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix 
    # Output - a set of 2D points in an array with size (N,2)
    size = len(X) # number of points
    
    X = np.hstack((X, np.ones((size,1))))
    Y_hat = X @ H.T
    Y = Y_hat / Y_hat[:,2:3]
    return Y[:,:2]


def fit_homography(XY):
    # TODO
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form [x^T_i,y^T_i] 1×4
    # Output - a 3*3 homography matrix
    X = XY[:,0:2]
    Y = XY[:,2:4]
    size = len(X) # number of points
    X = np.hstack((X, np.ones((size,1)))) 
    Y = np.hstack((Y, np.ones((size,1))))
    A = []
    for i in range(size):
        l1 = np.concatenate(([0,0,0],-1*X[i],Y[i][1]*X[i]))
        l2 = np.concatenate((X[i],[0,0,0],-1*Y[i][0]*X[i]))
        A.append(l1)
        A.append(l2)
    A = np.vstack(A)
    eigenValues, eigenVectors = np.linalg.eig(A.T @ A)
    idx = eigenValues.argsort()[::-1] # descending 
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    v = eigenVectors[:,-1] # the eignvector of A^TA with the smallest eignvalue
    # print(v)
    H = np.reshape(v, (3,3), order='C')
    # print(H)
    return H


def p1():
    # 1.2.3 - 1.2.5
    # TODO
    # 1. load points X from p1/transform.npy
    data = np.load('p1/transform.npy')
    X = data[:,0:2]
    Y = data[:,2:4] # b for ||Av-b||^2
    size = len(X)
    X_stack = np.hstack((X, np.ones((size,1)))) 

    # X augmented has a form a little bit complex. Refer to
    # Page 42 of Lecture8.pdf from CSE803 lectures.
    # TODO 
    # load X in a style of A in the ||Av-b||^2
    

    # 2. fit a transformation y=Sx+t, refer to numpy.linalg.lstsq
    W, _, _, _ = least_square(X_stack,Y,rcond=None)
    print(W) # fit X_stack @ W = Y

    # 3. transform the points 
    Y_hat = X_stack @ W
    print(Y_hat.shape) # 50 points

    # 4. plot the original points and transformed points
    # plot x, y and y_hat
    print(X.shape)
    print(Y.shape)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(X[:,0], X[:,1], marker='o', color='black', label='X')
    ax1.scatter(Y[:,0], Y[:,1], marker='s', color='red', label='Y')
    ax1.scatter(Y_hat[:,0], Y_hat[:,1], marker='s', color='green', label='Y_hat')
    plt.legend()
    plt.savefig('XandY.png')
    plt.close()

    # 1.2.6 - 1.2.8
    case = 8 # you will encounter 8 different transformations
    for i in range(case):
        XY = np.load('p1/points_case_'+str(i)+'.npy')
        # 1. generate your Homography matrix H using X and Y
        #
        #    specifically: fill function fit_homography() 
        #    such that H = fit_homography(XY)
        H = fit_homography(XY)
        # 2. Report H in your report
        print(H)
        # 3. Transform the points using H
        #
        #    specifically: fill function homography_transform
        #    such that Y_H = homography_transform(X, H)
        Y_H = homography_transform(XY[:,:2], H)
        # 4. Visualize points as three images in one figure
        # the following codes plot figure for you
        plt.scatter(XY[:,1],XY[:,0],c="red") #X
        plt.scatter(XY[:,3],XY[:,2],c="green") #Y
        plt.scatter(Y_H[:,1],Y_H[:,0],c="blue") #Y_hat
        plt.savefig('./case_'+str(i))
        plt.close()

def normalized_corr(C1,C2):
    C1 = C1 / np.linalg.norm(C1)
    C2 = C2 / np.linalg.norm(C2)
    return np.dot(C1.reshape(-1),C2.reshape(-1))

def stitchimage(imgleft, imgright):
    
    # TODO
    # 1. extract descriptors from images
    #    you may use SIFT/SURF of opencv
    sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    keypoints_sift_left, descriptors_left = sift.detectAndCompute(imgleft, None)
    # keypoints_surf, descriptors = surf.detectAndCompute(imgleft, None)
    imgleft_sift = cv2.drawKeypoints(imgleft, keypoints_sift_left, None)
    # imgleft_surf = cv2.drawKeypoints(imgleft, keypoints_surf, None)
    save_img('imgleft_sift.png',imgleft_sift)
    # save_img('imgleft_surf.png',imgleft_surf)

    keypoints_sift_right, descriptors_right = sift.detectAndCompute(imgright, None)
    # keypoints_surf, descriptors = surf.detectAndCompute(imgright, None)
    imgright_sift = cv2.drawKeypoints(imgright, keypoints_sift_right, None)
    # imgright_surf = cv2.drawKeypoints(imgright, keypoints_surf, None)
    save_img('imgright_sift.png',imgright_sift)
    # save_img('imgright_surf.png',imgright_surf)

    # 1.1. compute distances
    keypoints_left_locs = np.array([[keypoint.pt[0],keypoint.pt[1]] for keypoint in keypoints_sift_left])
    keypoints_right_locs = np.array([[keypoint.pt[0],keypoint.pt[1]] for keypoint in keypoints_sift_right])

    distances = np.ones([len(descriptors_left),len(descriptors_right)])
    # print(distances.shape)
    ### Below too slow
    # for i, l in enumerate(descriptors_left):
    #     for j, r in enumerate(descriptors_right):
    #         distances[i,j] = normalized_corr(l,r)
    descriptors_left = ((descriptors_left - np.mean(descriptors_left, axis = 0, keepdims=True))/np.std(descriptors_left, axis = 0, keepdims=True))
    descriptors_right = ((descriptors_right - np.mean(descriptors_right, axis = 0, keepdims=True))/np.std(descriptors_right, axis = 0, keepdims=True))
    norm_left = np.sum(descriptors_left**2, axis = 1, keepdims = True)
    norm_right = np.sum(descriptors_right**2, axis = 1, keepdims = True)
    distances = (norm_left + norm_right.T - 2 * (descriptors_left @ descriptors_right.T))**0.5
    print(distances.shape)

    # 2. select paired descriptors
    min_distances = np.min(distances, axis = 1)
    # print(len(np.nonzero(min_distances<3)[0]))
    idx = np.nonzero(min_distances<3)[0]
    idy = np.argmin(distances, axis = 1)[idx]
    assert(len(idx) == len(idy))
    print(len(idx))

    XY = np.hstack((keypoints_left_locs[idx,:], keypoints_right_locs[idy,:]))
    

    # 3. run RANSAC to find a transformation
    #    matrix which has most innerliers

    best_dist = None # For evaluatng best choice of H
    bestH = None
    bestCount = -1 # For evaluating outliers 
    for iteration in range(400):
        # RANSAC
        id_random = np.random.randint(0,len(idx), size = 50) # similar to the setting of q1()
        H = fit_homography(XY[id_random,:])
        Y_h_hat = homography_transform(XY[:,:2], H)
        dist = np.linalg.norm(Y_h_hat - XY[:,2:], axis = 1)
        count = np.sum(dist < 20)
        if count > bestCount:
            bestH = H
            bestCount = count
            best_dist = dist
    
    print(bestCount)
    print(np.mean(best_dist[best_dist < 20]**2))

    match_idx = (best_dist < 20)
    keypoint_left = []
    keypoint_right = []
    nidx = np.nonzero(match_idx)[0]
    matchLtoR = []
    midx = 0
    print(len(nidx))
    for i in nidx:
        keypoint_left.append(cv2.KeyPoint(keypoints_left_locs[idx[i]][0],keypoints_left_locs[idx[i]][1],1))
        keypoint_right.append(cv2.KeyPoint(keypoints_right_locs[idy[i]][0],keypoints_right_locs[idy[i]][1],1))
        matchLtoR.append(cv2.DMatch(midx,midx,best_dist[i]))
        midx += 1
    print(len(matchLtoR))
    
    match = None
    match = cv2.drawMatches(imgleft,keypoint_left,imgright,keypoint_right,matchLtoR, match )
    save_img('match.png', match)

    # 4. warp one image by your transformation 
    #    matrix
    #
    #    Hint: 
    #    a. you can use opencv to warp image
    #    b. Be careful about final image size
    
    bestH = np.array([[1,0,700],[0,1,300],[0,0,1]]) @ bestH
    warped = cv2.warpPerspective(imgleft, bestH, [2000,1000])
    

    # 5. combine two images, use average of them
    #    in the overlap area
    right = cv2.warpPerspective(imgright, np.array([[1,0,700],[0,1,300],[0,0,1]])@np.eye(3), [2000,1000])
    img = (warped.astype(np.int32) +right.astype(np.int32))
    overlap = np.logical_and(warped,right)
    img[overlap]  -= (0.5*img[overlap]).astype(np.int32)
    save_img('warp.png', warped)
    save_img('stitch.png', img)
    return bestH

def p2(p1, p2, savename):
    # read left and right images
    imgleft = read_img(p1)
    imgright = read_img(p2)
    save_img('imgleft.png',imgleft)
    save_img('imgright.png',imgright)

    
    
    # stitch image
    bestH = stitchimage(imgleft, imgright)
    imgleft = read_colorimg(p1)
    imgright = read_colorimg(p2)
    warped = cv2.warpPerspective(imgleft, bestH, [2000,1000])
    

    # 5. combine two images, use average of them
    #    in the overlap area
    right = cv2.warpPerspective(imgright, np.array([[1,0,700],[0,1,300],[0,0,1]])@np.eye(3), [2000,1000])
    img = (warped.astype(np.int32) +right.astype(np.int32))
    overlap = np.logical_and(warped,right)
    img[overlap]  -= (0.5*img[overlap]).astype(np.int32)
    save_img('warp.png', warped)
    save_img('stitch.png', img)
    # save stitched image
    


if __name__ == "__main__":
    # Problem 1
    # p1() # I comment this due to the run for p2

    # Problem 2
    # p2('p2/uttower_left.jpg', 'p2/uttower_right.jpg', 'uttower')
    p2('p2/bbb_left.jpg', 'p2/bbb_right.jpg', 'bbb')

