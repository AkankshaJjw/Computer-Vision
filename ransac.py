import numpy as np
import math
from proj2_code.least_squares_fundamental_matrix import solve_F
from proj2_code import two_view_data
from proj2_code import fundamental_matrix


def calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct):
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success #P
    -   sample_size: int the number of samples included in each RANSAC iteration #k
    -   ind_prob_success: float the probability that each element in a sample is correct #p
    
    Returns:
    -   num_samples: int the number of RANSAC iterations needed #S

    """
    # 1 - P = (1 - p^k) ^ S

    P = prob_success
    k = sample_size
    p = ind_prob_correct
    num = np.log(1-P)
    den = np.log(1 - (p)**k)
    num_samples = num/den

    ##############################
    # TODO: Student code goes here

    # raise NotImplementedError
    ##############################

    return num_samples


def find_inliers(x_0s, F, x_1s, threshold):
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the magnitude of the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    inliers = []
    N = len(x_0s)
    for i in range(N):
        #print("LOL1",x_1s[i])
        x1 = [x_1s[i,0],x_1s[i,1],1]
        x0 = [x_0s[i,0],x_0s[i,1],1]
        #print("LOL2",x1)
        line = np.dot(F, x1)
        dist = fundamental_matrix.point_line_distance(line, x0)
        dist = abs(dist)
        if(dist<=threshold):
            inliers.append(i)
    inliers = np.asarray(inliers)
    ##############################
    # TODO: Student code goes here

    # raise NotImplementedError
    ##############################

    return inliers


def ransac_fundamental_matrix(x_0s, x_1s):
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """
    P = 0.99
    k = 9
    p = 0.9
    max_inliers=0
    for i in range(14):
        x_ind = np.random.choice(range(len(x_0s)), 9)
        x0s_chosen= x_0s[x_ind,:]
        x1s_chosen= x_1s[x_ind,:]
        F = solve_F(x0s_chosen,x1s_chosen)
        inds = find_inliers(x_0s, F, x_1s, 1)
        num_inliers = len(inds)
        if num_inliers >= max_inliers:
            max_inliers = num_inliers
            best_F = F
            inliears_x_0 = x_0s[inds,:]
            inliears_x_1 = x_1s[inds,:]


    ##############################
    # TODO: Student code goes here

    # raise NotImplementedError
    ##############################

    return best_F, inliears_x_0, inliears_x_1
