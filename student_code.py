import numpy as np
import torch
from scipy import stats

from proj5_code.feature_matching.SIFTNet import get_siftnet_features
from proj5_code.utils import generate_sample_points


def pairwise_distances(X, Y):
    """
    This method will be very similar to the pairwise_distances() function found
    in sklearn (https://scikit-learn.org/stable/modules/generated/sklearn
    .metrics.pairwise_distances.html)
    However, you are NOT allowed to use any library functions like this
    pairwise_distances or pdist from scipy to do the calculation!

    The purpose of this method is to calculate pairwise distances between two
    sets of vectors. The distance metric we will be using is 'euclidean',
    which is the square root of the sum of squares between every value.
    (https://en.wikipedia.org/wiki/Euclidean_distance)

    Useful functions:
    -   np.linalg.norm()

    Args:
    -   X: N x d numpy array of d-dimensional features arranged along N rows
    -   Y: M x d numpy array of d-dimensional features arranged along M rows

    Returns:
    -   D: N x M numpy array where d(i, j) is the distance between row i of
    X and
        row j of Y
    """
    N, d_y = X.shape
    M, d_x = Y.shape
    assert d_y == d_x

    # D is the placeholder for the result
    D = np.zeros((N,M))
    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################
    for i in range(N):
        for j in range(M):
            vec1 = X[i]
            vec2 = Y[j]
            dis = np.linalg.norm(vec1-vec2)
            D[i,j] = dis
    #############################################################################
    #                             END OF YOUR CODE
    #                             #
    #############################################################################
    return D


def nearest_neighbor_classify(train_image_feats,
                              train_labels,
                              test_image_feats,
                              k=3):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which can increase the
    performance.

    Useful functions:
    -   D = pairwise_distances(X, Y) computes the distance matrix D between
    all pairs of rows in X and Y. This is the method you implemented above.
        -  X is a N x d numpy array of d-dimensional features arranged along
        N rows
        -  Y is a M x d numpy array of d-dimensional features arranged along
        N rows
        -  D is a N x M numpy array where d(i, j) is the distance between
        row i of X and row j of Y
    - np.argsort()
    - scipy.stats.mode()

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality
    of the feature representation
    -   train_labels: N element list, where each entry is a string
    indicating the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality
    of the feature representation.
    -   k: the k value in kNN, indicating how many votes we need to check
    for the label

    Returns:
    -   pred_labels: M element list, where each entry is a string indicating
    the predicted category for each testing image
    """
    print("test arr shape", test_image_feats.shape)
    pred_labels = []
    dist = pairwise_distances(train_image_feats, test_image_feats) #each col is for 1 test datapoint
    sorted_dist = np.argsort(dist, axis=0)
    print("sorted_dist\n", sorted_dist)
    print("k\n", k)
    sorted_dist2 = sorted_dist[0:k, :]
    print("sorted_dist2\n", sorted_dist2)
    mode_dist = stats.mode(sorted_dist2, axis=0)[0][0]
    print("mode_dist\n", mode_dist)
    print("labels\n", train_labels)
    pred_labels = np.array(train_labels)[mode_dist.astype(int)]
    pred_labels= list(pred_labels)
    print("pred_labels\n", pred_labels)
    #np.array(mode_dist)
    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return pred_labels


def kmeans(feature_vectors, k, max_iter=100):
    """
    Implement the k-means algorithm in this function. Initialize your centroids
    with random *unique* points from the input data, and repeat over the
    following process:
    1. calculate the distances from data points to the centroids
    2. assign them labels based on the distance - these are the clusters
    3. re-compute the centroids from the labeled clusters

    Please note that you are NOT allowed to use any library functions like
    vq.kmeans from scipy or kmeans from vlfeat to do the computation!

    Useful functions:
    -   np.random.randint
    -   np.linalg.norm
    -   pairwise_distances - implemented above
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (
    N, d)
            where N is the number of features and d is the dimensionality of
            the features
    -   k: the number of centroids to generate, of type int
    -   max_iter: the total number of iterations for k-means to run, of type
    int

    Returns:
    -   centroids: the generated centroids for the input feature_vectors,
    a Numpy
            array of shape (k, d)
    """

    # dummy centroids placeholder
    centroids = None
    np.random.seed(42)
    unique_feature_vectors = np.unique(feature_vectors, axis=0)
    initial_centroids = np.random.permutation(unique_feature_vectors.shape[0])[:k]
    centroids = unique_feature_vectors[initial_centroids]
    
    for j in range(max_iter):
        dist_to_centroid =  pairwise_distances(feature_vectors, centroids)
        cluster_labels = np.argmin(dist_to_centroid, axis = 1)
        for i in range(k):
            if (len(np.array(feature_vectors[cluster_labels == i]) == 0)): # if nothing is assigned to the cluster
                break
            centroids[i] = np.array(feature_vectors[cluster_labels == i]).mean(axis = 0)
                # print("\ncentroid i\n", centroids[i])
    centroids = np.array(centroids, dtype= np.float32)
    #############################################################################
    # TODO: YOUR CODE HERE
    #  #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #                             #
    #############################################################################
    print("centroids", centroids.shape, "\n", centroids)
    return centroids


def build_vocabulary(image_arrays, vocab_size=50, stride=20, max_iter=10):
    """
    This function will generate the vocabulary which will be further used
    for bag of words classification.

    To generate the vocab you first randomly sample features from the
    training set. Get SIFT features for the images using
    get_siftnet_features() method. This method takes as input the image
    tensor and x and y coordinates as arguments.
    Now cluster sampled features from all images using kmeans method
    implemented by you previously and return the vocabulary of words i.e.
    cluster centers.

    Points to note:
    *   To save computation time, you don't necessarily need to
    sample from all images, although it would be better to do so.
    *   Sample the descriptors from each image to save memory and
    speed up the clustering.
    *   For testing, you may experiment with larger
    stride so you just compute fewer points and check the result quickly.
    *   The default vocab_size of 50 is sufficient for you to get a
    decent accuracy (>40%), but you are free to experiment with other values.

    Useful functions:
    -   torch.from_numpy(img_array) for converting a numpy array to a torch
    tensor for siftnet
    -   torch.view() for reshaping the torch tensor
    -   use torch.type() or np.array(img_array).astype() for typecasting
    -   generate_sample_points() for sampling interest points

    Args:
    -   image_arrays: list of images in Numpy arrays, in grayscale
    -   vocab_size: size of vocabulary
    -   stride: the stride of your SIFT sampling

    Returns:
    -   vocab: This is a (vocab_size, dim) Numpy array (vocabulary). Where
    dim is the length of your SIFT descriptor. Each row is a cluster
    center/visual word.
    """

    dim = 128  # length of the SIFT descriptors that you are going to compute.
    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################
    all_fvs = []
    for img in image_arrays:
        height, width = img.shape
        xs, ys = generate_sample_points(height,width,stride)
        img_tensor = torch.from_numpy(img)
        #img_tensor = img_tensor.type(torch.FloatTensor)
        img_tensor = img_tensor.type(torch.float32)
        img_tensor = img_tensor.view(1,1,height, width)
        #print("hi1.5", img_tensor.size())
        this_fv = get_siftnet_features(img_tensor, xs, ys)
        #print("this_fv shape", this_fv.shape)
        all_fvs.extend(this_fv)
        #print("all_fv shape", np.array(all_fvs).shape)
    print("hi2")
    all_fvs = np.array(all_fvs)
    #print("all_fvs shape", all_fvs.shape, "\n", all_fvs)
    all_fvs = np.reshape(all_fvs, (-1,dim) )
    #print("all_fvs", all_fvs.shape, all_fvs)
    vocab = kmeans(all_fvs, k= vocab_size, max_iter= 100)
    vocab = np.array(vocab, dtype= np.float32)
    print("hi3")
    #############################################################################
    #                             END OF YOUR CODE
    #                             #
    #############################################################################

    return vocab


def kmeans_quantize(raw_data_pts, centroids):
    """
    Implement the k-means quantization in this function. Given the input
    data and the centroids, assign each of the data entry to the closest
    centroid.

    Useful functions:
    -   pairwise_distances
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (
    N, d) where N is the number of input data, and d is the dimension of it,
    given the standard SIFT descriptor, d  = 128
    -   centroids: the generated centroids for the input feature_vectors,
    a Numpy
            array of shape (k, D)

    Returns:
    -   indices: the index of the centroid which is closest to the data points,
            a Numpy array of shape (N, )

    """
    dist_to_centroid =  pairwise_distances(raw_data_pts, centroids)
    indices = np.argmin(dist_to_centroid, axis = 1)
    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #                             #
    #############################################################################
    return indices


def get_bags_of_sifts(image_arrays, vocabulary, stride=5):
    """
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   torch.from_numpy(img_array) for converting a numpy array to a torch
    tensor for siftnet
    -   torch.view() for reshaping the torch tensor
    -   use torch.type() or np.array(img_array).astype() for typecasting
    -   generate_sample_points() for sampling interest points
    -   get_siftnet_features() from SIFTNet: you can pass in the image
    tensor in grayscale, together with the sampled x and y positions to
    obtain the SIFT features
    -   np.histogram() : easy way to help you calculate for a particular
    image, how is the visual words span across the vocab
    -   np.linalg.norm() for normalizing the histogram


    Args:
    -   image_arrays: A list of N PIL Image objects
    -   vocabulary: A numpy array of dimensions: vocab_size x 128 where each
    row is a kmeans centroid or visual word.
    -   stride: same functionality as the stride in build_vocabulary().

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
    feature representation. In this case, d will equal the number of
    clusters or equivalently the number of entries in each image's histogram
    (vocab_size) below.
    """
    # load vocabulary
    vocab = vocabulary
    vocab_size = len(vocab)
    num_images = len(image_arrays)
    feats = np.zeros((num_images, vocab_size))
    feats_og = np.zeros((num_images, vocab_size))
    
    for ind in range(num_images):
        img = image_arrays[ind]
        h,w = img.shape
        xs, ys = generate_sample_points(h,w, stride)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.view((1,1,h,w))
        img_tensor = img_tensor.type(torch.float32)
        ft= get_siftnet_features(img_tensor, xs, ys)
        assgn = kmeans_quantize(ft, vocabulary)
        hist,_ = np.histogram(assgn, bins=range(vocab_size+1), normed=False)
        feats_og[ind] = hist
        hist = hist/np.linalg.norm(hist, ord=2)
        feats[ind] = hist
    #############################################################################
    # TODO: YOUR CODE HERE
    #  #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    #                             #
    #############################################################################

    return feats
