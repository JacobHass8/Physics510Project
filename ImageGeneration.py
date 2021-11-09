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

    centers : list of tuples
        x, y coordinates of simulated particles

    Examples
    --------
    >>> wave = 0.5
    >>> NA = 0.9
    >>> scale = 0.01
    >>> SNR = 10
    >>> camera_scale = 0.1
    >>> B = 1
    >>> img = multiple_part_img(wave, NA, scale, camera_scale, SNR, B, 11, [(-4, -4), (4, 4)])
    >>> plt.imshow(img, 'gray')
    '''

    bin_scaling = camera_scale / scale
    zero = np.zeros(shape=(int(size*bin_scaling), int(size*bin_scaling)))
    for x, y in centers:
        img = PSF(size * bin_scaling, wave, NA, scale, x * bin_scaling, y * bin_scaling) * SNR ** 2
        zero += img
    binned_image = bin_image(zero, scale, camera_scale) + B * bin_scaling
    return np.random.poisson(binned_image)

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

    centers : list of list of tuples
        x, y coordinates of simulated particles

    Examples
    --------
    paths = diffusing_paths(centers=[(5,5), (-5, -5)], size=50, step_size=0.5)
    wave = 0.5
    NA = 0.9
    scale = 0.01
    SNR = 10
    camera_scale = 0.1
    B = 1
    size = 30
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

    # Create stack of images
    img_shape = (size, size, len(centers))
    img_stack = np.zeros(img_shape)

    for i, center in enumerate(centers):
        img = multiple_part_img(wave, NA, scale, camera_scale, SNR, B, size, center)
        img_stack[:, :, i] = img
    return img_stack

def diffusing_path(center=(0,0), size=50, step_size=0.1):
    '''
    Generating a particle diffusing in 2D.

    Parameters
    ----------
    center : tuple
        Initial starting position of particle as (x, y)

    size : int
        Number of time steps to run through

    step_size : float
        Step size for each time

    Returns
    -------
    path : 2D numpy array
        First column is x-values and second column is y-values
    '''

    # High=2 b/c it is exclusive for whatever reason
    steps_x = np.random.randint(low=-1, high=2, size=size) * step_size
    steps_y = np.random.randint(low=-1, high=2, size=size) * step_size
    position_x = np.cumsum(steps_x) + center[0]
    position_y = np.cumsum(steps_y) + center[1]

    return np.array((position_x, position_y)).T

def diffusing_paths(centers=[(0, 0)], size=50, step_size=0.1):
    '''
    Generate multiple diffusing paths in 2D.

    Parameters
    ----------
    centers : list of tuples
        Starting position of each particle as (x, y) coordinates.

    size : int
        Number of time steps to run through

    step_size : float
        Step size for each time

    Returns
    -------
    path : list of list of tuples
        Particles coordinates at each time
    '''

    paths = [[] for _ in range(size)]
    for cent in centers:
        path = diffusing_path(cent, size, step_size)
        path = list(map(tuple, path))
        for j, pair in enumerate(path):
            paths[j].append(pair)
    return paths

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
