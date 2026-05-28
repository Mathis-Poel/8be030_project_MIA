"""
Utility functions for segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import segmentation as seg
from scipy import ndimage


def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    g = np.zeros_like(x, dtype=float)

    for k in range(x.size):
        x_forward = x.copy()
        x_backward = x.copy()

        x_forward.flat[k] += h
        x_backward.flat[k] -= h

        g.flat[k] = (fun(x_forward) - fun(x_backward)) / (2 * h)

    return g


def scatter_data(X, Y, feature0=0, feature1=1, ax=None):
    k = 1000
    if len(X) > k:
        idx = np.random.randint(len(X), size=k)
        X = X[idx, :]
        Y = Y[idx]

    class_labels, indices1, indices2 = np.unique(Y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == i
        lbl = 'X, class ' + str(class_labels[i])
        ax.scatter(X[idx2, feature0], X[idx2, feature1], color=c, label=lbl)

    return ax


def create_dataset(image_number, slice_number, task):
    X, feature_labels = extract_features(image_number, slice_number)
    Y = create_labels(image_number, slice_number, task)

    return X, Y, feature_labels


def extract_features(image_number, slice_number):
    # Let op: pas deze map aan als jouw data ergens anders staat.
    base_dir = '../data/dataset_brains/'

    t1 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t1.tif')
    t2 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t2.tif')

    features = ()

    t1_float = t1.astype(float)
    t2_float = t2.astype(float)

    t1f = t1_float.flatten().T.reshape(-1, 1)
    t2f = t2_float.flatten().T.reshape(-1, 1)

    X = np.concatenate((t1f, t2f), axis=1)

    features += ('T1 intensity',)
    features += ('T2 intensity',)

    # Extra feature 1: distance to image center.
    coord_f, coord_im = seg.extract_coordinate_feature(t1)

    # Extra features 2 and 3: smoothed intensities.
    t1_smooth = ndimage.gaussian_filter(t1_float, sigma=2)
    t2_smooth = ndimage.gaussian_filter(t2_float, sigma=2)

    # Extra features 4 and 5: gradient magnitude / edge strength.
    t1_grad_x = ndimage.sobel(t1_float, axis=0)
    t1_grad_y = ndimage.sobel(t1_float, axis=1)
    t1_edge = np.sqrt(t1_grad_x ** 2 + t1_grad_y ** 2)

    t2_grad_x = ndimage.sobel(t2_float, axis=0)
    t2_grad_y = ndimage.sobel(t2_float, axis=1)
    t2_edge = np.sqrt(t2_grad_x ** 2 + t2_grad_y ** 2)

    t1_smooth_f = t1_smooth.flatten().T.reshape(-1, 1)
    t2_smooth_f = t2_smooth.flatten().T.reshape(-1, 1)
    t1_edge_f = t1_edge.flatten().T.reshape(-1, 1)
    t2_edge_f = t2_edge.flatten().T.reshape(-1, 1)

    X = np.concatenate(
        (X, coord_f, t1_smooth_f, t2_smooth_f, t1_edge_f, t2_edge_f),
        axis=1
    )

    features += ('Distance to center',)
    features += ('Smoothed T1 intensity',)
    features += ('Smoothed T2 intensity',)
    features += ('T1 edge strength',)
    features += ('T2 edge strength',)

    return X, features


def create_labels(image_number, slice_number, task):
    base_dir = '../data/dataset_brains/'

    I = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_gt.tif')

    if task == 'brain':
        Y = I > 0
    elif task == 'tissue':
        white_matter = np.isin(I, [2, 5])
        gray_matter = np.isin(I, [3, 7])
        csf = np.isin(I, [4, 8])
        background = np.isin(I, [0, 1, 6])

        Y = np.copy(I)
        Y[background] = 0
        Y[white_matter] = 1
        Y[gray_matter] = 2
        Y[csf] = 3
    else:
        print(task)
        raise ValueError("Variable 'task' must be one of two values: 'brain' or 'tissue'")

    Y = Y.flatten().T
    Y = Y.reshape(-1, 1)

    return Y


def dice_overlap(true_labels, predicted_labels, smooth=1.):
    # Returns the Dice coefficient for two binary label vectors.
    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten().astype(bool)
    p = predicted_labels.flatten().astype(bool)

    intersection = np.sum(t & p)
    dice = (2.0 * intersection + smooth) / (np.sum(t) + np.sum(p) + smooth)

    return dice


def dice_multiclass(true_labels, predicted_labels):
    all_classes, indices1, indices2 = np.unique(true_labels, return_index=True, return_inverse=True)

    dice_score = np.empty((len(all_classes), 1))
    dice_score[:] = np.nan

    for i in np.arange(len(all_classes)):
        temp_true = true_labels.copy()
        temp_true[true_labels == all_classes[i]] = 1
        temp_true[true_labels != all_classes[i]] = 0

        temp_predicted = predicted_labels.copy()
        temp_predicted[predicted_labels == all_classes[i]] = 1
        temp_predicted[predicted_labels != all_classes[i]] = 0

        dice_score[i] = dice_overlap(temp_true.astype(int), temp_predicted.astype(int))

    dice_score_mean = dice_score.mean()

    return dice_score_mean


def classification_error(true_labels, predicted_labels):
    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    err = np.mean(t != p)

    return err
