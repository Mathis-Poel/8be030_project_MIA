"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import registration as reg
from IPython.display import display, clear_output


def intensity_based_registration_rigid():

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('../data/image_data/3_1_t1.tif')
    Im = plt.imread('../data/image_data/3_1_t2.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these

    
    x = np.array([0.,0.,0.])
    

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_xdata(iterations[:k+1])
        learning_curve.set_ydata(similarity[:k+1])
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw()
        display(fig)
def intentsity_based_registration_affine():
    I = plt.imread('../data/image_data/3_1_t1.tif')
    Im = plt.imread('../data/image_data/3_1_t2.tif')

    # Needs 7 parameters [rot, scaling_x, scaling_y,
    # shearing_x, shearing_y, translation_x, translation_y]
    # with the NOTE above it gives a x_affine of
    x_affine = np.array([0., 1., 1.,0.,0.,0.,0.]) 
    
