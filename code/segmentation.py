"""
Segmentation module main code.
"""

import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans



# SECTION 1. Segmentation in feature space


def generate_gaussian_data(N=100, mu1=[0, 0], mu2=[2, 0], sigma1=[[1, 0], [0, 1]], sigma2=[[1, 0], [0, 1]]):
    # Generates a 2D toy dataset with 2 classes, N samples per class. 
    # Class 1 is Gaussian distributed with mu1 and sigma2
    # Class 2 is Gaussian distributed with mu2 and sigma2.
    # Input:
    # N - Number of samples per class (2N in total)
    # mu1 - 1x2 vector, mean of class 1
    # mu2 - 1x2 vector, mean of class 2
    # sigma1 - 2x2 matrix, covariance of class 1
    # sigma2 - 2x2 matrix, covariance of class 2
    
    # Generate class 1
    # Rotate data according to covariance matrix (must be positive
    # definite), and add the mean
    A = np.linalg.cholesky(sigma1)
    data1 = np.random.randn(N,2).dot(A) + mu1

    # Generate class 2
    B = np.linalg.cholesky(sigma2)
    data2 = np.random.randn(N,2).dot(B) + mu2
    
    # Put the data together
    X = np.concatenate((data1, data2), axis=0)

    # Create labels
    Y = np.concatenate((np.zeros((N,1)), np.ones((N,1))), axis=0)

    return X, Y


def extract_coordinate_feature(im):
    # Creates a coordinate feature, which encodes how far a pixel is
    # from the center of the image.
    n_rows, n_cols = im.shape

    x_center = np.floor(n_rows / 2)
    y_center = np.floor(n_cols / 2)

    ar = np.arange(n_cols).reshape(1, -1)
    x_coord = np.tile(ar, (n_rows, 1))
    ar = ar.T
    y_coord = np.tile(ar, (1, n_cols))

    # Distance of every pixel to the image center.
    coord_im = np.sqrt((x_coord - y_center) ** 2 + (y_coord - x_center) ** 2)

    # Normalize to range [0, 1], so it has comparable scale.
    if np.max(coord_im) != 0:
        coord_im = coord_im / np.max(coord_im)

    c = coord_im.flatten().T
    c = c.reshape(-1, 1)

    return c, coord_im


def normalize_data(train_data, test_data=None):
    mean_train = np.mean(train_data, axis=0)
    std_train = np.std(train_data, axis=0)

    # Avoid division by zero for constant features.
    std_train[std_train == 0] = 1

    train_data = train_data - mean_train
    train_data = train_data / std_train

    if test_data is not None:
        test_data = test_data - mean_train
        test_data = test_data / std_train

    return train_data, test_data


def cost_kmeans(X, w_vector):
    # Computes the cost of assigning data in X to clusters in w_vector.
    n, m = X.shape
    K = int(len(w_vector) / m)
    W = w_vector.reshape(K, m)

    D = scipy.spatial.distance.cdist(X, W, metric='euclidean')
    min_dist = np.min(D, axis=1)
    J = np.sum(min_dist ** 2)

    return J


def kmeans_clustering_sklearn(test_data, K=4):
    kmeans = KMeans(
        n_clusters=K,
        random_state=0,
        n_init=10
    )

    pred = kmeans.fit_predict(test_data)

    # labels sorteren op gemiddelde intensiteit van feature 0
    centers = kmeans.cluster_centers_
    sorted_order = np.argsort(centers[:, 0])

    pred_sorted = np.zeros_like(pred)

    for new_label, old_label in enumerate(sorted_order):
        pred_sorted[pred == old_label] = new_label

    return pred_sorted


def nn_classifier(train_data, train_labels, test_data):
    # 1-NN is the same as k-NN with k=1.
    predicted_labels = knn_classifier(train_data, train_labels, test_data, k=1)
    return predicted_labels


def knn_classifier(train_data, train_labels, test_data, k):
    D = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean')
    sort_ix = np.argsort(D, axis=1)
    sort_ix_k = sort_ix[:, :k]
    predicted_labels = train_labels[sort_ix_k]
    predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]

    return predicted_labels


# SECTION 2. Generalization and overfitting


def mypca(X):
    # Rotates the data X such that dimensions of X_pca are uncorrelated
    # and sorted by variance.
    X = X - np.mean(X, axis=0)

    # Covariance matrix of the features.
    C = np.cov(X, rowvar=False)

    # Eigenvalues/eigenvectors. eigh is used because covariance is symmetric.
    w, v = np.linalg.eigh(C)

    # Sort from largest to smallest eigenvalue.
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]

    # Rotate data to PCA space.
    X_pca = X @ v

    fraction_variance = np.zeros((X_pca.shape[1], 1))
    for i in np.arange(X_pca.shape[1]):
        fraction_variance[i] = np.sum(w[:i + 1]) / np.sum(w)

    return X_pca, v, w, fraction_variance


def segmentation_combined_knn(train_data_matrix, train_labels_matrix, test_data, k=1):
    r, c = train_labels_matrix.shape

    predicted_labels = np.empty([r, c])
    predicted_labels[:] = np.nan

    for i in np.arange(c):
        predicted_labels[:, i] = segmentation_knn(
            train_data_matrix[:, :, i],
            train_labels_matrix[:, i],
            test_data,
            k
        )

    predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]

    return predicted_labels.astype(bool)


def segmentation_knn(train_data, train_labels, test_data, k=1):
    # Subsample training data for efficiency.
    # num_samples = 3000
    num_samples = 57000
    ix = np.random.randint(train_data.shape[0], size=num_samples)

    subset_train_data = train_data[ix, :]
    subset_train_labels = train_labels[ix]

    train_data_norm, test_data_norm = normalize_data(subset_train_data, test_data)

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_data_norm, subset_train_labels.ravel())
    predicted_labels = neigh.predict(test_data_norm)

    return predicted_labels
