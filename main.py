import math
from typing import List, Tuple

import numpy
import numpy as np
import pandas as pd
import scipy.optimize as opt
from numpy import all, array, asarray, exp, where
from pandas import DataFrame
from skimage.feature import peak_local_max
from skimage.filters import gaussian as gaussian_filter


def compute(im: numpy.ndarray, options: dict) -> tuple:
    """
    Compute the Point Spread Function (PSF) from an image.

    :param im: Input image.
    :type im: numpy.ndarray
    :param options: Dictionary of options for the computation.
    :type options: dict
    :return: DataFrame containing PSF data and the smoothed image.
    :rtype: tuple
    """
    im = im.astype(float)

    beads, maxima, centers, smoothed = getCenters(im, options)

    if options["max_beads"]:
        max_beads = options["max_beads"]
    else:
        max_beads = len(beads)

    if options["fit_mode"] == "3D":

        initial_guess = (
            0,  # x0
            0,  # y0
            0,  # z0
            (options["wavelength_nm"] / 1000.0 / (2 * options["NA"]))  # sigma_x
            * options["px_per_um_lat"]
            / (4 * numpy.sqrt(-0.5 * numpy.log(0.5)))
            + options["bead_size_um"],
            (options["wavelength_nm"] / 1000.0 / (2 * options["NA"]))  # sigma_y
            * options["px_per_um_lat"]
            / (4 * numpy.sqrt(-0.5 * numpy.log(0.5)))
            + options["bead_size_um"],
            (options["wavelength_nm"] / 1000.0 / (2 * options["NA"]))  # sigma_z
            * options["px_per_um_ax"]
            / (4 * numpy.sqrt(-0.5 * numpy.log(0.5)))
            + options["bead_size_um"],
            1,  # amplitude
            0,  # offset
            0,  # rotx
            0,  # roty
            0,  # rotz
        )

        lower_bounds = (
            -beads[0].shape[2] / 2.0,  # x0
            -beads[0].shape[1] / 2.0,  # y0
            -beads[0].shape[0] / 2.0,  # z0
            0.1,  # sigma_x
            0.1,  # sigma_y
            0.1,  # sigma_z
            0.5,  # amplitude
            -0.1,  # offset
            -math.pi / 2.0,  # rotx
            -math.pi / 2.0,  # roty
            -math.pi / 2.0,  # rotz
        )

        upper_bounds = (
            beads[0].shape[2] / 2.0,  # x0
            beads[0].shape[1] / 2.0,  # y0
            beads[0].shape[0] / 2.0,  # z0
            beads[0].shape[2],  # sigma_x
            beads[0].shape[1],  # sigma_y
            beads[0].shape[0],  # sigma_z
            1.1,  # amplitude
            0.1,  # offset
            math.pi / 2.0,  # rotx
            math.pi / 2.0,  # roty
            math.pi / 2.0,  # rotz
        )

        x = numpy.linspace(
            -beads[0].shape[2] / 2.0, beads[0].shape[2] / 2.0, beads[0].shape[2]
        )
        y = numpy.linspace(
            -beads[0].shape[1] / 2.0, beads[0].shape[1] / 2.0, beads[0].shape[1]
        )
        z = numpy.linspace(
            -beads[0].shape[0] / 2.0, beads[0].shape[0] / 2.0, beads[0].shape[0]
        )

        X, Z, Y = numpy.meshgrid(x, z, y)

        data = [
            get3DPSF(
                i,
                numpy.array([X, Y, Z]),
                initial_guess,
                lower_bounds,
                upper_bounds,
                options,
            )
            for i in beads[:max_beads]
        ]
    else:

        initial_guess = (
            0,  # x0
            (options["wavelength_nm"] / 1000.0 / (2 * options["NA"]))  # sigma
            * options["px_per_um_lat"]
            / (4 * numpy.sqrt(-0.5 * numpy.log(0.5)))
            + options["bead_size_um"],
            1,  # amplitude
            0,  # offset
        )

        lower_bounds = (
            -beads[0].shape[2] / 2.0,  # x0
            0.1,  # sigma
            0.5,  # amplitude
            -0.1,  # offset
        )

        upper_bounds = (
            beads[0].shape[2] / 2.0,  # x0
            beads[0].shape[0],  # sigma
            1.1,  # amplitude
            0.1,  # offset
        )

        data = [
            get2DPSF(
                i,
                initial_guess,
                lower_bounds,
                upper_bounds,
                options,
            )
            for i in beads[:max_beads]
        ]

    PSF = pd.concat([i for i in data])
    PSF["max"] = maxima[:max_beads]
    PSF["x_center_um"] = centers[:max_beads, 2]
    PSF["y_center_um"] = centers[:max_beads, 1]
    PSF["z_center_um"] = centers[:max_beads, 0]
    PSF["bead"] = beads[:max_beads]
    PSF = PSF.reset_index().drop(["index"], axis=1)

    return PSF, smoothed


def inside(shape: tuple, center: tuple, window: tuple) -> bool:
    """
    Check if the center is inside the given shape with the specified window.

    :param shape: Shape of the image.
    :type shape: tuple
    :param center: Center coordinates.
    :type center: tuple
    :param window: Window size.
    :type window: tuple
    :return: True if the center is inside the shape, False otherwise.
    :rtype: bool
    """
    return all(
        [
            (center[i] - window[i] >= 0) & (center[i] + window[i] <= shape[i])
            for i in range(0, 3)
        ]
    )


def volume(im: numpy.ndarray, center: tuple, window: tuple) -> numpy.ndarray:
    """
    Extract a volume from the image centered at the given coordinates.

    :param im: Input image.
    :type im: numpy.ndarray
    :param center: Center coordinates.
    :type center: tuple
    :param window: Window size.
    :type window: tuple
    :return: Extracted volume.
    :rtype: numpy.ndarray
    """
    if inside(im.shape, center, window):
        volume = im[
            (center[0] - window[0]):(center[0] + window[0]),
            (center[1] - window[1]):(center[1] + window[1]),
            (center[2] - window[2]):(center[2] + window[2]),
        ]
        volume = volume.astype("float64")
        baseline = volume[[0, -1], [0, -1], [0, -1]].mean()
        volume = volume - baseline
        volume = volume / volume.max()
        return volume


def findBeads(im: numpy.ndarray, window: tuple, thresh: float) -> tuple:
    """
    Find beads in the image using local maxima detection.

    :param im: Input image.
    :type im: numpy.ndarray
    :param window: Window size.
    :type window: tuple
    :param thresh: Threshold for peak detection.
    :type thresh: float
    :return: Coordinates of detected beads and the smoothed image.
    :rtype: tuple
    """
    smoothed = gaussian_filter(image=im, sigma=1.0, mode="nearest", cval=0)
    centers = peak_local_max(
        smoothed, min_distance=3, threshold_rel=thresh, exclude_border=True
    )
    return centers, smoothed.max(axis=0)


def keepBeads(
    im: numpy.ndarray, window: tuple, centers: numpy.ndarray, options: dict
) -> numpy.ndarray:
    """
    Filter out beads that are too close to each other or outside the image boundaries.

    :param im: Input image.
    :type im: numpy.ndarray
    :param window: Window size.
    :type window: tuple
    :param centers: Coordinates of detected beads.
    :type centers: numpy.ndarray
    :param options: Dictionary of options for the computation.
    :type options: dict
    :return: Filtered coordinates of beads.
    :rtype: numpy.ndarray
    """
    centersM = asarray(
        [
            [
                x[0] / options["px_per_um_lat"],
                x[1] / options["px_per_um_lat"],
                x[2] / options["px_per_um_lat"],
            ]
            for x in centers
        ]
    )
    centerDists = [nearest(x, centersM) for x in centersM]
    keep = where([x > 3 for x in centerDists])
    centers = centers[keep[0], :]
    keep = where([inside(im.shape, x, window) for x in centers])
    return centers[keep[0], :]


def getCenters(im: numpy.ndarray, options: dict) -> tuple:
    """
    Get the centers of beads in the image.

    :param im: Input image.
    :type im: numpy.ndarray
    :param options: Dictionary of options for the computation.
    :type options: dict
    :return: Beads, maxima, centers, and the smoothed image.
    :rtype: tuple
    """
    window = [
        options["window_um"][0] * options["px_per_um_ax"],
        options["window_um"][1] * options["px_per_um_lat"],
        options["window_um"][2] * options["px_per_um_lat"],
    ]
    window = [round(x) for x in window]
    centers, smoothed = findBeads(im, window, options["thresh"])
    centers = keepBeads(im, window, centers, options)
    beads = [volume(im, x, window) for x in centers]
    maxima = [im[x[0], x[1], x[2]] for x in centers]
    return beads, maxima, centers, smoothed


def getSlices(average: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the lateral and axial profiles from the average array.

    :param average: 3D array of averaged values.
    :type average: numpy.ndarray
    :return: Tuple containing the lateral and axial profiles.
    :rtype: tuple
    """
    latProfile = (
        average.mean(axis=0).mean(axis=1) + average.mean(axis=0).mean(axis=1)
    ) / 2
    axProfile = (
        average.mean(axis=1).mean(axis=1) + average.mean(axis=2).mean(axis=1)
    ) / 2
    return latProfile, axProfile


def dist(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points, excluding the first dimension.

    :param x: First point.
    :type x: numpy.ndarray
    :param y: Second point.
    :type y: numpy.ndarray
    :return: Euclidean distance between x and y.
    :rtype: float
    """
    return ((x - y) ** 2)[1:].sum() ** (0.5)


def nearest(x: np.ndarray, centers: np.ndarray) -> float:
    """
    Find the nearest center to the given point x.

    :param x: Point to find the nearest center for.
    :type x: numpy.ndarray
    :param centers: Array of center points.
    :type centers: numpy.ndarray
    :return: Distance to the nearest center.
    :rtype: float
    """
    z = [dist(x, y) for y in centers if not (x == y).all()]
    return abs(array(z)).min(axis=0)


def get3DPSF(
    bead: np.ndarray,
    XYZ: np.ndarray,
    initial_guess: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
    options: dict,
) -> DataFrame:
    """
    Fit a 3D Gaussian to the bead data.

    :param bead: 3D array representing the bead.
    :type bead: numpy.ndarray
    :param XYZ: Meshgrid array for the coordinates.
    :type XYZ: numpy.ndarray
    :param initial_guess: Initial guess for the Gaussian parameters.
    :type initial_guess: list
    :param lower_bounds: Lower bounds for the Gaussian parameters.
    :type lower_bounds: list
    :param upper_bounds: Upper bounds for the Gaussian parameters.
    :type upper_bounds: list
    :param options: Dictionary of options for the computation.
    :type options: dict
    :return: DataFrame containing the fitted parameters.
    :rtype: pandas.DataFrame
    """
    bead = bead / np.max(bead)

    popt, _ = opt.curve_fit(
        gaussian_3D,
        XYZ,
        bead.ravel(),
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
    )

    xo, yo, zo, sigma_x, sigma_y, sigma_z, amplitude, offset, rotx, roty, rotz = (
        popt[0],
        popt[1],
        popt[2],
        popt[3],
        popt[4],
        popt[5],
        popt[6],
        popt[7],
        popt[8],
        popt[9],
        popt[10],
    )

    FWHM_x = (
        np.abs(4 * sigma_x * np.sqrt(-0.5 * np.log(0.5))) / options["px_per_um_lat"]
    )
    FWHM_y = (
        np.abs(4 * sigma_y * np.sqrt(-0.5 * np.log(0.5))) / options["px_per_um_lat"]
    )
    FWHM_z = np.abs(4 * sigma_z * np.sqrt(-0.5 * np.log(0.5))) / options["px_per_um_ax"]

    data = DataFrame(
        [xo, yo, zo, amplitude, offset, FWHM_x, FWHM_y, FWHM_z, rotx, roty, rotz],
        index=[
            "xo",
            "yo",
            "zo",
            "amplitude",
            "offset",
            "FWHM_x",
            "FWHM_y",
            "FWHM_z",
            "rotx",
            "roty",
            "rotz",
        ],
    ).T
    return data


def get2DPSF(
    bead: np.ndarray,
    initial_guess: dict,
    lower_bounds: dict,
    upper_bounds: dict,
    options: dict,
) -> DataFrame:
    """
    Fit 2D Gaussians to the lateral and axial profiles of the bead.

    :param bead: 3D array representing the bead.
    :type bead: numpy.ndarray
    :param initial_guess: Initial guess for the Gaussian parameters.
    :type initial_guess: list
    :param lower_bounds: Lower bounds for the Gaussian parameters.
    :type lower_bounds: list
    :param upper_bounds: Upper bounds for the Gaussian parameters.
    :type upper_bounds: list
    :param options: Dictionary of options for the computation.
    :type options: dict
    :return: DataFrame containing the fitted parameters.
    :rtype: pandas.DataFrame
    """
    bead_lateral, bead_axial = getSlices(bead)

    # lateral fit
    x = array(range(bead_lateral.shape[0]))
    popt_lateral, _ = opt.curve_fit(
        gaussian_1D,
        x,
        bead_lateral,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
    )
    amplitude, xo, sigma_lateral, offset = (
        popt_lateral[0],
        popt_lateral[1],
        popt_lateral[2],
        popt_lateral[3],
    )
    FWHM_lateral = (
        np.abs(4 * sigma_lateral * np.sqrt(-0.5 * np.log(0.5)))
        / options["px_per_um_lat"]
    )

    # axial fit
    x = array(range(bead_axial.shape[0]))
    popt_axial, _ = opt.curve_fit(
        gaussian_1D,
        x,
        bead_axial,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
    )
    amplitude, zo, sigma_axial, offset = (
        popt_axial[0],
        popt_axial[1],
        popt_axial[2],
        popt_axial[3],
    )
    FWHM_axial = (
        np.abs(4 * sigma_axial * np.sqrt(-0.5 * np.log(0.5))) / options["px_per_um_ax"]
    )

    data = DataFrame(
        [xo, xo, zo, amplitude, offset, FWHM_lateral, FWHM_axial, 0, 0, 0],
        index=[
            "xo",
            "yo",
            "zo",
            "amplitude",
            "offset",
            "FWHM_lat",
            "FWHM_ax",
            "rotx",
            "roty",
            "rotz",
        ],
    ).T
    return data


def gaussian_1D(
    x: np.ndarray, a: float, mu: float, sigma: float, b: float
) -> np.ndarray:
    """
    1D Gaussian function.

    :param x: Input array.
    :type x: numpy.ndarray
    :param a: Amplitude of the Gaussian.
    :type a: float
    :param mu: Mean of the Gaussian.
    :type mu: float
    :param sigma: Standard deviation of the Gaussian.
    :type sigma: float
    :param b: Baseline offset.
    :type b: float
    :return: Gaussian function evaluated at x.
    :rtype: numpy.ndarray
    """
    return a * exp(-((x - mu) ** 2) / (2 * sigma**2)) + b


def gaussian_3D(
    XYZ: np.ndarray,
    xo: float,
    yo: float,
    zo: float,
    sigma_x: float,
    sigma_y: float,
    sigma_z: float,
    amplitude: float,
    offset: float,
    rotx: float,
    roty: float,
    rotz: float,
) -> np.ndarray:
    """
    3D Gaussian function with rotation.

    :param XYZ: Meshgrid array for the coordinates.
    :type XYZ: numpy.ndarray
    :param xo: X-coordinate of the center.
    :type xo: float
    :param yo: Y-coordinate of the center.
    :type yo: float
    :param zo: Z-coordinate of the center.
    :type zo: float
    :param sigma_x: Standard deviation along the X-axis.
    :type sigma_x: float
    :param sigma_y: Standard deviation along the Y-axis.
    :type sigma_y: float
    :param sigma_z: Standard deviation along the Z-axis.
    :type sigma_z: float
    :param amplitude: Amplitude of the Gaussian.
    :type amplitude: float
    :param offset: Baseline offset.
    :type offset: float
    :param rotx: Rotation around the X-axis.
    :type rotx: float
    :param roty: Rotation around the Y-axis.
    :type roty: float
    :param rotz: Rotation around the Z-axis.
    :type rotz: float
    :return: Flattened 3D Gaussian function evaluated at XYZ.
    :rtype: numpy.ndarray
    """
    XRot = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rotx), np.sin(rotx)],
            [0, -np.sin(rotx), np.cos(rotx)],
        ]
    )

    YRot = np.array(
        [
            [np.cos(roty), 0, -np.sin(roty)],
            [0, 1, 0],
            [np.sin(roty), 0, np.cos(roty)],
        ]
    )

    ZRot = np.array(
        [
            [np.cos(rotz), np.sin(rotz), 0],
            [-np.sin(rotz), np.cos(rotz), 0],
            [0, 0, 1],
        ]
    )

    XYZ = np.einsum("ij,jabc->iabc", XRot, XYZ)
    XYZ = np.einsum("ij,jabc->iabc", YRot, XYZ)
    XYZ = np.einsum("ij,jabc->iabc", ZRot, XYZ)

    g = offset + amplitude * np.exp(
        -(
            ((XYZ[0] - xo) ** 2) / (2 * sigma_x**2)
            + ((XYZ[1] - yo) ** 2) / (2 * sigma_y**2)
            + ((XYZ[2] - zo) ** 2) / (2 * sigma_z**2)
        )
    )

    return g.ravel()
