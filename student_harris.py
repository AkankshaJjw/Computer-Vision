import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.
    
    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """
    
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    kernel = cv2.getGaussianKernel(ksize, sigma) #returns ksize x 1 matrix
    kernel_t = kernel.transpose() # 1 x ksize matrix
    kernel = np.dot(kernel, kernel_t)
    #############################################################################
    
    # raise NotImplementedError('`get_gaussian_kernel` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return kernel

def my_filter2D(image, filt, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   filt: filter that will be used in the convolution

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    # conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    m, n = filt.shape
    r,c = image.shape
    k = (int)((m-1)/2)
    image = cv2.copyMakeBorder(image, top=k, bottom=k, left=k, right=k, borderType= cv2.BORDER_CONSTANT, value=0)
    # print("k: ", k)
    conv_image = np.zeros((r,c))
    for i in range(k, r+k):
        for j in range(k, c+k):
            roi = image[i-k:i+k+1, j-k:j+k+1]
            part = (roi*filt).sum() + bias
            conv_image[i-k, j-k] = part

    #print("my ans\n", conv_image)
    #############################################################################

    # raise NotImplementedError('`my_filter2D` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """
    
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    #change the following to use my_filter2D
    # ix =  cv2.filter2D(image, ddepth =-1, kernel = sobel_x, borderType = cv2.BORDER_CONSTANT)
    # iy = cv2.filter2D(image, ddepth=-1, kernel = sobel_y, borderType = cv2.BORDER_CONSTANT)
    ix = my_filter2D(image, sobel_x)
    iy = my_filter2D(image, sobel_y)
    # print("my ix:\n", ix)
    # print("my iy:\n", iy)
    #############################################################################

    # raise NotImplementedError('`get_gradients` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 32):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: A numpy array of shape (N,) containing the x coordinate of each pixel
    -   y: A numpy array of shape (N,) containing the y coordinate of each pixel
    -   c: A numpy array of shape (N,) containing the confidences of each pixel
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        (set this to 16 for unit testing). treat the center point of this window as the bottom right
        of the center most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   c: numpy nd-array of dim  (N,) containing the confidences of each pixel
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    new_x =[]
    new_y =[]
    new_c =[]
    m = image.shape[0]
    n = image.shape[1]
    ws_2 = (int)(window_size/2)
    for x1,y1,c1 in zip(x,y,c):
        if (((y1-ws_2) >= 0) & ((y1+ws_2-1) < m) & ((x1-ws_2) >= 0) & ((x1+ws_2-1) < n)):
            new_x.append(x1)
            new_y.append(y1)
            new_c.append(c1)

    #############################################################################

    # raise NotImplementedError('`remove_border_vals` function in ' +
    # '`student_harris.py` needs to be implemented')
    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return new_x, new_y, new_c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.

    Helpful functions: my_filter2D

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: A numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    ix2 = ix * ix
    iy2 = iy * iy
    ixiy = ix * iy
    # ix2 = np.square(ix)
    # iy2 = np.square(iy)
    # ixiy = np.multiply(ix,iy)
    gauss = get_gaussian_kernel(ksize, sigma)
    sx2 = my_filter2D(ix2, gauss)
    sy2 = my_filter2D(iy2, gauss)
    sxsy = my_filter2D(ixiy, gauss)
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################

    # raise NotImplementedError('`second_moments` function in ' +
    # '`student_harris.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments calculate corner resposne.
    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]


    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """
    m,n = sx2.shape
    R = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            M = [[sx2[i,j], sxsy[i,j]], [sxsy[i,j], sy2[i,j]]]
            this_R = np.linalg.det(M) - alpha*((np.trace(M))**2)
            R[i,j] = this_R
    #print("myR\n", R)

    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################

    # raise NotImplementedError('`corner_response` function in ' +
    # '`student_harris.py` needs to be implemented')
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non zero. We also do not want local
    maxima that are very small as well so remove all values that are below the median.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   ksize: int that is the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """
    max_filt = maximum_filter(R, size = neighborhood_size, mode='constant')
    print("max filt:\n", max_filt)
    filtm, filtn = max_filt.shape
    m,n = R.shape
    print("max_filt_size\n", filtm, filtn)
    print("non padded R size\n", m, n)
    padm = (int)((filtm-m)/2)
    padn = (int)((filtn-n)/2)
    med = np.median(R)
    R[R<med] = 0
    R = cv2.copyMakeBorder(R, top=padm, bottom=padm, left=padn, right=padn, borderType= cv2.BORDER_CONSTANT, value=0)
    print("padded R size:\n", R.shape)
    R_local_pts = np.zeros((m,n))
    r = (neighborhood_size-1)/2
    for i in range(padm, m+padm):
        for j in range(padn, n+padn):
            local = max_filt[i,j]
            if (R[i,j]==local):
               R_local_pts[i-padm,j-padn] = local
            else:
               R_local_pts[i-padm,j-padn] = 0 
    print("R_local_pts:\n", R_local_pts)
    
    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################

    # raise NotImplementedError('`non_max_suppression` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts
    

def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer of number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences: numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """
    x, y, R_local_pts, confidences = None, None, None, None
    
    x1=[]
    y1=[]
    c1=[]
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    ix, iy = get_gradients(image)
    sx2, sy2, sxsy = second_moments(ix, iy)
    R = corner_response(sx2, sy2, sxsy, 0.05)
    R_local_pts = non_max_suppression(R)

    m, n = R_local_pts.shape
    for i in range(m): #rows
        for j in range(n): #cols
            if (R_local_pts[i,j] != 0):
                y1.append(i)
                x1.append(j)
                c1.append(R_local_pts[i,j])
    x2,y2,c2 = remove_border_vals(image, x1,y1,c1)
    inds = np.argsort(c2)[0:n_pts]
    x=[]
    y=[]
    confidences=[]
    for z in inds:
        x.append(x2[z])
        y.append(y2[z])
        confidences.append(c2[z])
    


    #############################################################################
    
    # raise NotImplementedError('`get_interest_points` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    x = np.array(x)
    y = np.array(y)
    #print("MyR\n", R_local_pts)
    return x,y, R_local_pts, confidences


