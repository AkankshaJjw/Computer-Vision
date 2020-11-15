import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq

import time
#print("works")

def objective_func(x, **kwargs):
    """
    Calculates the difference in image (pixel coordinates) and returns 
    it as a 2*n_points vector

    Args: 
    -        x: numpy array of 11 parameters of P in vector form 
                (remember you will have to fix P_34=1) to estimate the reprojection error
    - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                retrieve these 2D and 3D points and then use them to compute 
                the reprojection error.
    Returns:
    -     diff: A N_points-d vector (1-D numpy array) of differences betwen 
                projected and actual 2D points

    """
    #print(len(x))
    list2d = kwargs.get('pts2d')
    list3d = kwargs.get('pts3d')
    num_pts = len(list2d)
    x = np.append(x,1)
    # for i in x:
    #     i = i/x[11]
    #print(x)
    diff=np.zeros(2*num_pts)
    P_hat = np.reshape(x, (3,4))
    for i in range(num_pts):
        x_iw = np.append(list3d[i],1) #3D coords
        proj_2d = np.dot(P_hat, x_iw)
        for j in range(len(proj_2d)):
            proj_2d[j] = proj_2d[j]/proj_2d[2] #to make homogeneous
        x_i = list2d[i] #actual2D coords
        diff[2*i] = proj_2d[0] - x_i[0]
        diff[2*i+1] = proj_2d[1] - x_i[1]
        
    
    

    ##############################
    # TODO: Student code goes here

    # raise NotImplementedError
    ##############################

    return diff


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                       or n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    projected_points_2d = np.zeros((len(points_3d),2))
    for i in range(len(projected_points_2d)):
        row = projected_points_2d[i]
        row[0] = (P[0][0]*points_3d[i][0] + P[0][1]*points_3d[i][1] + P[0][2]*points_3d[i][2] + P[0][3]) / (P[2][0]*points_3d[i][0] + P[2][1]*points_3d[i][1] + P[2][2]*points_3d[i][2] + P[2][3])
        row[1] = (P[1][0]*points_3d[i][0] + P[1][1]*points_3d[i][1] + P[1][2]*points_3d[i][2] + P[1][3]) / (P[2][0]*points_3d[i][0] + P[2][1]*points_3d[i][1] + P[2][2]*points_3d[i][2] + P[2][3])
        #print(row)
    ##############################
    # TODO: Student code goes here

    # raise NotImplementedError
    ##############################
    return projected_points_2d


def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''
    start_time = time.time()

    ##############################
    # TODO: Student code goes here
    kwargs = {'pts2d':pts2d,
          'pts3d':pts3d}
    initial_guess1 = initial_guess.flatten()[:11]
    print(len(initial_guess1))
    P = least_squares(fun=objective_func, x0=initial_guess1, method='lm', max_nfev=50000, verbose=2, kwargs=kwargs).x
    P = np.append(P,1)
    P = np.reshape(P,(3,4))
    # raise NotImplementedError
    ##############################

    print("Time since optimization start", time.time() - start_time)
    #print("blahhhh P", P)
    return P


def decompose_camera_matrix(P: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    #print("P is", P)
    M = np.delete(P,3,1) #delete 4th column of P
    #print("dimensions",len(M.shape))
    K,R = rq(M)
    ##############################
    # TODO: Student code goes here

    # raise NotImplementedError
    ##############################

    return K, R


def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    KR = np.dot(K,R_T) # P = K R_T [I | -t]
    KR_inv = np.linalg.inv(KR)
    final1 = np.dot(KR_inv,P) # = [I | -t] which is 3x4 matrix
    cc = -final1[:,3] #only keep t

    ##############################
    # TODO: Student code goes here

    # raise NotImplementedError
    ##############################

    return cc
