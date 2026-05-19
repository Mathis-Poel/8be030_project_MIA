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
    I = plt.imread('../data/image_data/3_2_t1.tif')
    Im = plt.imread('../data/image_data/3_2_t2.tif')

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
    mu = 0.0005 #iets kleiner dan 0.001, anders divergeert het in lokaal minimum

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


def compare_ncc_mi_registration():
    # Fixed = T1, Moving = T2
    I = plt.imread('../data/image_data/3_2_t1.tif')
    Im = plt.imread('../data/image_data/3_2_t2.tif')

    # affine parameters:
    # [rotation, sx, sy, shearx, sheary, tx, ty]
    x_ncc = np.array([0., 1., 1., 0., 0., 0., 0.])
    x_mi  = np.array([0., 1., 1., 0., 0., 0., 0.])

    fun_ncc = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
    fun_mi  = lambda x: reg.affine_mi(I, Im, x, return_transform=False)

    mu_ncc = 0.0001
    mu_mi = 0.0005

    num_iter = 61 

    sim_ncc = []
    sim_mi = []

    for k in range(num_iter):
        sim_ncc.append(fun_ncc(x_ncc))
        sim_mi.append(fun_mi(x_mi))

        g_ncc = reg.ngradient(fun_ncc, x_ncc)
        g_mi  = reg.ngradient(fun_mi, x_mi, h=1e-2)  # alleen dit veranderd

        x_ncc += mu_ncc * g_ncc
        x_mi  += mu_mi  * g_mi

    # final transformed images
    _, Im_ncc, _ = reg.affine_corr(I, Im, x_ncc)
    _, Im_mi, _ = reg.affine_mi(I, Im, x_mi)

    # visualization
    fig = plt.figure(figsize=(15,8))

    ax1 = fig.add_subplot(231)
    ax1.imshow(I, cmap='gray')
    ax1.set_title("Fixed T1")

    ax2 = fig.add_subplot(232)
    ax2.imshow(I, cmap='gray')
    ax2.imshow(Im_ncc, cmap='hot', alpha=0.5)
    ax2.set_title("NCC Registration")

    ax3 = fig.add_subplot(233)
    ax3.imshow(I, cmap='gray')
    ax3.imshow(Im_mi, cmap='hot', alpha=0.5)
    ax3.set_title("MI Registration")

    ax4 = fig.add_subplot(212)
    ax4.plot(sim_ncc, label='NCC')
    ax4.plot(sim_mi, label='MI')
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Similarity")
    ax4.legend()
    ax4.grid()

    plt.tight_layout()
    plt.show()
