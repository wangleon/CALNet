import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import csv
import numpy as np
from astropy.io import fits
from scipy.interpolate import InterpolatedUnivariateSpline

def get_time_flux_from_filepath(filePath):
    '''
    get time and flux from file path of lc file
    Args:
        filePath: full path of lc file.
    returns:
        t: time series.
        f: flux series.
    '''
    table = fits.getdata(filePath)
    time = table['TIME']
    flux = table['PDCSAP_FLUX']
    q_lst = table['QUALITY']
    m = q_lst == 0
    time = time[m]
    flux = flux[m]
    m2 = ~np.isnan(flux)
    t = time[m2]
    f = flux[m2]
    return t,f


def lc_split(filename,gap_width):
    """
    Obtain the light curve from the file name and split it based on its gaps.
    Args:
        filename: filename of lc file.
        gap_width: when the gap reaches the gap_width, we split the light curve.
    Returns:
        out_time: The split time series in a two-dimensional list format.
        out_flux: The split flux series in a two-dimensional list format.
    """
    
    # get time and flux
    all_time, all_flux = get_time_flux_from_filepath(filename)
    
    # Handle single-segment inputs.
    if isinstance(all_time, np.ndarray) and all_time.ndim == 1:
        all_time = [all_time]
        all_flux = [all_flux]

    out_time = []
    out_flux = []
    for time, flux in zip(all_time, all_flux):
        start = 0
        for end in range(1, len(time) + 1):
            # Choose the largest endpoint such that time[start:end] has no gaps.
            if end == len(time) or time[end] - time[end - 1] > gap_width:
                out_time.append(time[start:end])
                out_flux.append(flux[start:end])
                start = end

    return out_time, out_flux

def dropMax(t,f,ratio):
    '''
    drop the top 1% of the maximum values of light curve.
    Args:
        t: One-dimensional time series
        f: One-dimensional light curve
        ratio: The proportion of data that we need to retain.
    Returns:
        t: time series after drop 1% max
        f: flux series after drop 1% max
    
    '''
    mask = f < np.percentile(f, ratio)
    f = f[mask]
    t = t[mask]
    return t,f

def robust_mean(y, cut):
    """
    Computes a robust mean estimate in the presence of outliers.
    Args:
        y: 1D numpy array. Assumed to be normally distributed with outliers.
        cut: Points more than this number of standard deviations from the median are ignored.
    Returns:
        mean: A robust estimate of the mean of y.
        mean_stddev: The standard deviation of the mean.
        mask: Boolean array with the same length as y. Values corresponding to outliers in y are False. All other values are True.
    """
    # First, make a robust estimate of the standard deviation of y, assuming y is
    # normally distributed. The conversion factor of 1.4826 takes the median
    # absolute deviation to the standard deviation of a normal distribution.
    # See, e.g. https://www.mathworks.com/help/stats/mad.html.
    absdev = np.abs(y - np.median(y))
    sigma = 1.4826 * np.median(absdev)

    # If the previous estimate of the standard deviation using the median absolute
    # deviation is zero, fall back to a robust estimate using the mean absolute
    # deviation. This estimator has a different conversion factor of 1.253.
    # See, e.g. https://www.mathworks.com/help/stats/mad.html.
    if sigma < 1.0e-24:
        sigma = 1.253 * np.mean(absdev)

    # Identify outliers using our estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Now, recompute the standard deviation, using the sample standard deviation
    # of non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers. The
    # following formula is an approximation, see
    # http://w.astro.berkeley.edu/~johnjohn/idlprocs/robust_mean.pro.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    # Identify outliers using our second estimate of the standard deviation of y.
    mask = absdev <= cut * sigma

    # Now, recompute the standard deviation, using the sample standard deviation
    # with non-outlier points.
    sigma = np.std(y[mask])

    # Compensate the estimate of sigma due to trimming away outliers.
    sc = np.max([cut, 1.0])
    if sc <= 4.5:
        sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    # Final estimate is the sample mean with outliers removed.
    mean = np.mean(y[mask])
    mean_stddev = sigma / np.sqrt(len(y) - 1.0)

    return mean, mean_stddev, mask

def process_lc(time,flux,outlier_cut,flux_len):
    '''
    Preprocess the light curve.
    Args:
        time: One-dimensional time series
        flux: One-dimensional light curve
        outlier_cut: Points more than this number of standard deviations from the median are ignored.
        flux_len: the length of the light curve we set
    Returns:
        f: preprocessed light curve.
    
    '''
    # Normalize the time to the range [0, 1].
    t_min = np.min(time)
    t_max = np.max(time)
    time = (time - t_min) / (t_max - t_min)

    #The mask is a boolean vector, where `TRUE` indicates the data points used for interpolation fitting.
    mask = np.ones_like(time, dtype=np.bool_)
    
    #Iteratively fit the spline.
    maxiter=3 #Number of iterations
    for _ in range(maxiter):
        time = time[mask]
        flux = flux[mask]
        spl = InterpolatedUnivariateSpline(time, flux)
        fit = spl(time)
        residuals = flux - fit
        new_mask = robust_mean(residuals, cut=outlier_cut)[2]
        # if np.all(new_mask == mask):
        #     break  # Spline converged.
        mask = new_mask
    
    #Generate simulated data using the fitted spline.
    x = np.linspace(0, 1, flux_len)
    fit_flux = spl(x)
    
    # Normalize the light curve.
    f_max = max(fit_flux)
    f_min = min(fit_flux)
    f = (fit_flux - f_min)/(f_max - f_min)
    
    return f 

def writeData(data,csv_file):
    # Create a new file if it does not exist.
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

    # Write row data.
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

