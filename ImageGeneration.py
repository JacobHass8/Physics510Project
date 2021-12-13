# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:02:19 2021

@author: jacob
"""

import numpy as np
from scipy.special import j1
from skimage.transform import downscale_local_mean
from matplotlib import pyplot as plt
from skimage import morphology as morph
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
#from numba import jit
import pandas as pd
from scipy.io import loadmat

def distance_matrix(size_x, size_y, xc=0, yc=0):
    '''
    Returns the distance from the center of a specified image
    size and center. Formula is regular Euclidian distance:

        sqrt(x^2 + y^2)

    Parameters
    ----------
    size_x, size_y : ints
        Size of the image in pixels

    xc, yc : floats
        Offset from the center pixel

    Returns
    -------
    distance : 2D numpy array
        Distance to the center of a 2D array
    '''

    x_range = np.arange((-size_x+1)/2, (size_x-1)/2+1, 1) + xc
    y_range = np.arange((-size_y+1)/2, (size_y-1)/2+1, 1) + yc
    xx, yy = np.meshgrid(x_range[::-1], y_range)
    distance = np.sqrt(xx**2 + yy**2)
    return distance

def PSF(N, wave, NA, scale, xc=0, yc=0):
    '''
    Calculate the  PSF for given inputs.

    Parameters
    ----------
    N : int
        Size of the image

    wave : float
        Wavelength

    NA : float
        Numerical aperature

    scale : float
        Distance / pixel of the image

    xc, yc : floats
        Center of the PSF. Should be between 0 and 1

    Returns
    -------
    img : NxN numpy array
        PSF calculated over the image
    '''
    r = distance_matrix(N, N, xc, yc) * scale
    v = (2 * np.pi / wave) * NA * r
    img = 4 * (j1(v) / v) ** 2

    # Need to set center point to 1 - it will be nan so just set nan to 1
    # Limit j1/x x-> 0 is 1/2
    img[np.isnan(img)] = 1
    return img

def bin_image(image, scale, camera_scale):
    '''
    Pixelate an image from an input scale to camera scale.

    Parameters
    ----------
    img : 2D numpy array
        Image to bin

    scale : float
        Original scale of the image

    camera_scale : float
        Resolution of camera

    Returns
    -------
    img : 2D numpy array
        Binned image (intensities are preserved)

    Note
    ----
    Scale and camera_scale should be integer multiples
    '''

    bin_size = camera_scale / scale
    assert bin_size > 1, "Camera scale must be larger than scale"

    int_bin_size = int(bin_size)
    assert int_bin_size == bin_size, "Camera scale and scale not integer multiples"

    img = downscale_local_mean(image, (int_bin_size, int_bin_size))
    return img

def SNR_Point(N, wave, NA, scale, SNR, camera_scale, B, xc=0, yc=0):
    '''
    Create a simulated point source with given parameters.

    Parameters
    ----------
    N : int
        Size of the image

    wave : float
        Wavelength

    NA : float
        Numerical aperature

    scale : float
        Distance / pixel of the image

    SNR : float
        Signal to noise ratio calculated with reference to brightest pixel

    camera_scale : float
        Resolution of camera

    B : float
        Background noise value (should probably be < SNR if you
        want a good image)

    xc, yc : floats (optional)
        Center of the PSF. Should be between 0 and 1

    Example
    -------
    >>> N = 150
    >>> wave = 0.5
    >>> NA = 0.9
    >>> scale = 0.01
    >>> SNR = 10
    >>> camera_scale = 0.1
    >>> B = 10
    >>> img2 = SNR_Point(N, wave, NA, scale, SNR, camera_scale, B, xc=0.3, yc=0.3)
    >>> plt.imshow(img2)
    '''

    bin_scaling = camera_scale / scale
    xc = xc * bin_scaling
    yc = yc * bin_scaling
    img = PSF(N, wave, NA, scale, xc, yc) * SNR ** 2
    binned_image = bin_image(img, scale, camera_scale)
    background = np.random.poisson(B, binned_image.shape)
    return np.random.poisson(binned_image) + background

def multiple_part_img(wave, NA, scale, camera_scale, SNR, B, size, centers):
    '''
    Simulate multiple particles in an image.

    Parameters
    ----------
    N : int
        Size of the image

    wave : float
        Wavelength

    NA : float
        Numerical aperature

    scale : float
        Distance / pixel of the image

    SNR : float
        Signal to noise ratio calculated with reference to brightest pixel

    camera_scale : float
        Resolution of camera

    B : float
        Background noise value (should probably be < SNR if you
        want a good image)

    size : int
        Size of ending image

    centers : 2D numpy array
        x, y coordinates of simulated particles where first column
        is x coordinates and second column is y coordinates

    Examples
    --------
    >>> wave = 0.5
    >>> NA = 0.9
    >>> scale = 0.01
    >>> SNR = 10
    >>> camera_scale = 0.1
    >>> B = 1
    >>> centers = np.array([[-4, -4], [4, 4]])
    >>> img = multiple_part_img(wave, NA, scale, camera_scale, SNR, B, 11, centers)
    >>> plt.imshow(img, 'gray')
    '''

    bin_scaling = camera_scale / scale
    zero = np.zeros(shape=(int(size*bin_scaling), int(size*bin_scaling)))
    for x, y in centers:
        img = PSF(size * bin_scaling, wave, NA, scale, x * bin_scaling, y * bin_scaling) * SNR ** 2
        zero += img
    binned_image = bin_image(zero, scale, camera_scale) + B * bin_scaling
    return np.random.poisson(binned_image)

def diffusing_path(center=None, size=50, step_size=0.1):
    '''
    Generating a particle diffusing in 2D.

    Parameters
    ----------
    center : list
        Initial starting position of particle as (x, y)

    size : int
        Number of time steps to run through

    step_size : float
        Step size for each time

    Returns
    -------
    path : 2D numpy array
        First column is x-values, second column is y-values and
        third column is time
    '''

    if center is None:
        center = [0, 0]

    # High=2 b/c it is exclusive for whatever reason
    steps_x = np.random.randint(low=-1, high=2, size=size) * step_size
    steps_y = np.random.randint(low=-1, high=2, size=size) * step_size
    position_x = np.cumsum(steps_x) + center[0]
    position_y = np.cumsum(steps_y) + center[1]
    time = range(0, len(position_x))

    return np.array((position_x, position_y, time)).T

def diffusing_paths(centers=None, size=50, step_size=0.1):
    '''
    Generate multiple diffusing paths in 2D.

    Parameters
    ----------
    centers : 2D numpy array
        Starting position of each particle as (x, y) coordinates.
        First column should be x-values and second column is y-values.

    size : int
        Number of time steps to run through

    step_size : float
        Step size for each time

    Returns
    -------
    path : pandas Dataframe
        Columns for x, y, time and particle id

    Examples
    --------
    >>> centers = np.array([[5, 5], [-5, -5]])
    >>> paths = diffusing_paths(centers)
    >>> for part_id in np.unique(paths[:, 3]):
    >>>     data = paths[paths[:, 3] == part_id]
    >>>     plt.plot(data[:, 0], data[:, 1])
    '''

    if centers is None:
        centers = np.array([[0, 0]])

    # going to make a fourth column for the id
    shape = (size * len(centers), 4)
    paths = np.zeros(shape)

    row_idx = 0
    for part_id, cent in enumerate(centers):
        path = diffusing_path(cent, size, step_size)
        paths[row_idx : row_idx + len(path), :3] = path
        paths[row_idx : row_idx + len(path), 3] = part_id
        row_idx += len(path)
    return pd.DataFrame(data=paths, columns=['x','y','time', 'id'])

def generate_imgs(wave, NA, scale, camera_scale, SNR, B, size, centers):
    '''
    Generate multiple images from given centers.

    Parameters
    ----------
    N : int
        Size of the image

    wave : float
        Wavelength

    NA : float
        Numerical aperature

    scale : float
        Distance / pixel of the image

    SNR : float
        Signal to noise ratio calculated with reference to brightest pixel

    camera_scale : float
        Resolution of camera

    B : float
        Background noise value (should probably be < SNR if you
        want a good image)

    size : int
        Size of ending image

    centers : pandas dataframe
        Should have columns x, y and time


    Examples
    --------
    centers = np.array([[0, 0],
                        [1, 1],
                        [2, 2],
                        [-1, -1],
                        [-2, -2],
                        [1, -1],
                        [-1, 1]]) * 6
    paths = diffusing_paths(centers=centers, size=100, step_size=0.5)
    wave = 0.5
    NA = 0.9
    scale = 0.01
    SNR = 10
    camera_scale = 0.1
    B = 1
    size = 50
    imgs = generate_imgs(wave, NA, scale, camera_scale, SNR, B, size, paths)
    for j in range(imgs.shape[2]):
        fig, ax = plt.subplots()
        img = imgs[:, :, j]
        x_centers, y_centers = find_multiple_centers(img)
        ax.imshow(img, 'gray')
        ax.scatter(x_centers, y_centers, c='r')
        fig.savefig(f"./Data/Frame{j}.png", bbox_inches='tight')
        plt.close(fig)
    '''

    sorted_centers = centers.sort_values(by=['time'])
    times = np.unique(sorted_centers['time'])

    img_shape = (size, size, len(times))
    img_stack = np.zeros(img_shape)
    for i, t in enumerate(times):
        current_centers = sorted_centers[sorted_centers['time'] == t]
        xy_vals = current_centers[['x', 'y']].values
        img = multiple_part_img(wave, NA, scale, camera_scale, SNR, B, size, xy_vals)
        img_stack[:, :, i] = img
    return img_stack

def find_bright_spots(img, size=2):
    """
    Find the particle regions in an image. Criteria is bright spot
    is 2 standard deviations above the median.

    Parameters
    ----------
    img : 2D array
        Image to find particle center estimates of

    Returns
    -------
    centers : list of tuples
        Index positions of particles

    Examples
    --------
    >>> wave = 0.5
    >>> NA = 0.9
    >>> scale = 0.01
    >>> SNR = 10
    >>> camera_scale = 0.1
    >>> B = 1
    >>> img = multiple_part_img(wave, NA, scale, camera_scale, SNR, 11, [(-4, -4), (4, 4)])
    >>> plt.imshow(img, 'gray')
    >>> bright = find_bright_spots(img)
    >>> plt.scatter(*bright, c='r')
    """

    med = np.median(img)
    std = np.std(img)
    dilated = morph.dilation(img, morph.disk(size))
    bright_spots = dilated == img
    bright_spots[bright_spots * img < (med + std*2)] = 0

    # Okay, so this returns a weird data structure. First column is x-vals and
    # second column is y-vals
    bright_coordinates = np.array(np.nonzero(bright_spots)).T
    return bright_coordinates

def objfun(args, x, y, z):
    '''
    Objective function to minimize for guassian distributed noise.
    '''
    x0, y0, A0, sigma, B = args
    guassprob = A0*np.exp(-(x-x0)**2/(2*sigma**2) - (y-y0)**2/(2*sigma**2)) + B
    Lk = z * np.log(guassprob) - guassprob
    return -np.sum(Lk)

def mle_center(img):
    '''
    Find the center of an image using MLE.

    Parameters
    ----------
    img : 2D numpy array
        Image should be narrowed down to only the particle region.

    Returns
    -------
    x0, y0 : ints
        Center of the point particle

    Examples
    --------
    >>> wave = 0.5
    >>> NA = 0.9
    >>> scale = 0.01
    >>> SNR = 10
    >>> camera_scale = 0.1
    >>> B = 1
    >>> img = multiple_part_img(wave, NA, scale, camera_scale, SNR, 11, [(1,3)])
    >>> plt.imshow(img, 'gray')
    >>> bright = mle_center(img)
    >>> plt.scatter(*bright, c='r')
    '''

    xx, yy = np.meshgrid(range(0, img.shape[0]), range(0, img.shape[1]))
    x, y = xx.flatten(), yy.flatten()
    z = img.flatten()

    bounds = ((min(x), max(x)), (min(y), max(y)), (0, max(z)*3/2), (0, None), (0, max(z)*3/2))
    x0 = (x[np.argmax(z)], y[np.argmax(z)], max(z), img.shape[0] / 3, 0)
    params = minimize(objfun, x0=x0, args=(x, y, z), bounds=bounds).x
    x0, y0 = params[0], params[1]
    return (x0, y0)

def find_multiple_centers(img, size=3):
    """
    Find the particle center for an image with multiple particles.

    Paramters
    ---------
    img : 2D numpy array
        Image to find particle centers in

    Returns
    -------
    x0, y0 : lists
        Particle centers measured from the top left hand corner of the image

    Examples
    --------
    >>> wave = 0.5
    >>> NA = 0.9
    >>> scale = 0.01
    >>> SNR = 10
    >>> camera_scale = 0.1
    >>> B = 1
    >>> img = multiple_part_img(wave, NA, scale, camera_scale, SNR, 20, [(1.5, 3.5), (-5, -5), (-5, 5), (5, -5)])
    >>> x_centers, y_centers = find_multiple_centers(img)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(img)
    >>> for x, y in zip(x_centers, y_centers):
    >>>     ax.scatter(x, y, c='r')
    >>> fig.savefig("Test.png")
    """

    bright_spots = find_bright_spots(img)

    x_centers = []
    y_centers = []
    for x, y in bright_spots:
        region= img[x-size:x+size+1, y-size:y+size+1]
        x0, y0 = mle_center(region)
        x0_offset = x0 - region.shape[0] // 2
        y0_offset = y0 - region.shape[1] // 2
        x_centers.append(x0_offset + y)
        y_centers.append(y0_offset + x)
    return x_centers, y_centers

#@jit()
def make_cost_matrix(initial_centers, final_centers):
    '''
    Make a cost matrix.

    Parameters
    ----------
    initial_centers : 2D numpy array
        First column is x-values and second column is y-values
        (x, y)

    final_centers : 2D numpy array
        First column is x-values and second column is y-values
        (x, y)

    Returns
    -------
    cost_matrix : 2D numpy array
        Matrix where cost_matrix[i, j] = msd(x_i, x_j) or
        is the mean squared displacement between the i and j
        particle.


    Examples
    --------
    >>> initial_centers = np.array([(0, 0), (1,1), (1, -1)])
    >>> final_centers =  np.array([(0.9, 0.8), (0.9, -1.1), (0.1, 0.1)])
    >>> cost_matrix = make_cost_matrix(initial_centers, final_centers)
    >>> print(cost_matrix)
    '''

    nParticles = len(initial_centers)

    assert len(initial_centers) == len(final_centers), "Must have the same number of particles"

    cost_matrix = np.zeros(shape=(nParticles, nParticles))
    for i in range(len(initial_centers)):
        for j in range(len(final_centers)):
            x_initial, y_initial = initial_centers[i, :]
            x_final, y_final = final_centers[j, :]
            msd = (x_initial - x_final) ** 2 + (y_initial - y_final) ** 2
            cost_matrix[i, j] = msd
    return cost_matrix

def assign_labels(paths):
    '''
    Assign labels to each of the centers based on the reported
    cost matrix minimization.

    Parameters
    ----------
    paths : pandas Dataframe
        Should have columns x, y, and time

    Returns
    -------
    paths : pandas Dataframe
        Same dataframe with new "particle_id" column

    Examples
    --------
    >>> centers = np.array([[0, 0],
                            [1, 1],
                            [2, 2],
                            [-1, -1],
                            [-2, -2],
                            [1, -1],
                            [-1, 1]]) * 6
    >>> paths = diffusing_paths(centers=centers, size=100, step_size=0.5)
    >>> df = assign_labels(paths)
    >>> print("Percent Correct: ", sum(df['particle_id'] == df['id']) / len(df) * 100)
    '''

    time_sorted_df = paths.sort_values(by=['time', 'id'])
    times = np.unique(time_sorted_df['time'])
    time_sorted_df['particle_id'] = range(len(time_sorted_df))

    for j in range(1, len(times)):
        current_time = time_sorted_df[time_sorted_df['time'] == times[j]]
        prev_time = time_sorted_df[time_sorted_df['time'] == times[j-1]]

        final_centers = current_time[['x', 'y']].values
        initial_centers = prev_time[['x', 'y']].values
        cost = make_cost_matrix(initial_centers, final_centers)
        row_idx, col_idx = linear_sum_assignment(cost)

        # note: row_idx will just be the range of the number of particles
        # so row_idx corresponds to initial particle value
        # and col_idx corresponds to final particle value

        for row, col in zip(row_idx, col_idx):
            prev_id = prev_time.iloc[row]['particle_id']
            time_sorted_df.loc[current_time.iloc[col].name, 'particle_id'] = prev_id

    return time_sorted_df
