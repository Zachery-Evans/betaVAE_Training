"""
This file contains functions for implementing beta-VAE model on new experimental IR spectra
Dependencies: pandas, numpy, matplotlib, sklearn, tensorflow
Author: Michael Grossutti
Date Created: 2023-04-01

"""
#imports
import pandas as pd 
import numpy as np 
import numpy.matlib 
from scipy.signal import find_peaks
from scipy import interpolate 
from scipy.signal import savgol_filter 
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy import sparse
from scipy.stats import norm
from scipy.integrate import simpson
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

def airpls(x, lam=100, porder=1, itermax=100):
    '''
    airpls.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
    Baseline correction using adaptive iteratively reweighted penalized least squares

    This program is a translation in python of the R source code of airPLS version 2.0
    by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls
    Reference:
    Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive
    iteratively reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

    Description from the original documentation:

    Baseline drift always blurs or even swamps signals and deteriorates analytical
    results, particularly in multivariate analysis.  It is necessary to correct
    baseline drift to perform further data analysis. Simple or modified polynomial
    fitting has been found to be effective in some extent. However, this method
    requires user intervention and prone to variability especially in low
    signal-to-noise ratio environments. The proposed adaptive iteratively
    reweighted Penalized Least Squares (airPLS) algorithm doesn't require any
    user intervention and prior information, such as detected peaks. It
    iteratively changes weights of sum squares errors (SSE) between the fitted
    baseline and original signals, and the weights of SSE are obtained adaptively
    using between previously fitted baseline and original signals. This baseline
    estimator is general, fast and flexible in fitting baseline.

    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        porder:
            integer indicating the order of the difference of penalties

    output:
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lam, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if(dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if(i == itermax):
                print('airpls: max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak,
        # so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
        
    return z


def WhittakerSmooth(x, w, lam, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x:
            input data (i.e. chromatogram of spectrum)
        w:
            binary masks (value of the mask is zero if a point belongs to peaks
            and one otherwise)
        lam:
            parameter that can be adjusted by user. The larger lambda is,  the
            smoother the resulting background
        differences:
            integer indicating the order of the difference of penalties

    output:
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
#   D = csc_matrix(np.diff(np.eye(m), differences))
    D = sparse.eye(m, format='csc')
    for i in range(differences):
        D = D[1:] - D[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = sparse.diags(w, 0, shape=(m, m))
    A = sparse.csc_matrix(W + (lam * D.T * D))
    B = sparse.csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)

def interpolate_spectrum(input_wavenumber, input_absorbance, low=898, high=3800):
    '''
    Arguments:
        input_wavenumber {array} -- a numpy array of shape [x,]
        input_absorbance {array} -- a numpy array of shape [x,]
        low {int} -- low wavenumber cutoff for interpolation window
        high {int} -- high wavenumumber cutoff for interpolation window   
        
    Returns:
        interpolated_wavenumber {array} -- interpolated wavenumber array as defined by low and high
        interpolated_absorbance {array} -- interpolated absorbance array
        
    '''

    tck = interpolate.make_splrep(input_wavenumber, input_absorbance, s=0)
    window = high - low + 1
    interpolated_wavenumber = np.linspace(low, high, window)
    interpolated_absorbance = interpolate.splev(interpolated_wavenumber, tck, der=0)
    
    return (interpolated_wavenumber, interpolated_absorbance)

def vector_normalization(spectrum):
    '''
    Arguments:
        spectrum {array} -- baseline corrected spectrum
    
    Outputs:
        spec_norm {array} -- vector normalized spectrum
    '''
    
    L2_norm = (sum(spectrum**2))**0.5
    
    spec_norm = spectrum / L2_norm
    
    return spec_norm

def minmax_normalization(spectrum):
    '''
    Min-Max 0 to 1 normalization on a per-spectrum basis.

    Arguments:
        spectrum {array} -- a numpy array of shape [x,]

    Returns:
        mix_max_norm {array} -- a numpy array of shape [x,], 0 to 1 normalized
    '''
    min_max_norm = (spectrum - spectrum.min()) / ( spectrum.max() -  spectrum.min()) 

    return min_max_norm

def integrate_peak(wavenumbers, absorbance, low=2000, high=2050):
    """
    Integrate the area of a peak in the absorbance array between the specified low
    and high wavenumber values using Simpson's rule.
    """
    # Find the indices of the low and high wavenumber values
    low_idx = (wavenumbers >= low).argmax()
    high_idx = (wavenumbers <= high).argmin()

    # Use Simpson's rule to integrate the peak
    integrated_area = simpson(absorbance[low_idx:high_idx+1], x=wavenumbers[low_idx:high_idx+1])

    return integrated_area

def rubberband_baseline(wavenumbers, absorbance):
    """
    Calculate rubberband baseline.
    
    Arguments:
        wavenumber {array} -- a numpy array 
        absorbance {array} -- a numpy array

    Returns:
        baseline {array} -- a numpy array
    """
    x = np.asarray(wavenumbers, dtype=np.float64)
    y = np.asarray(absorbance, dtype=np.float64)
    points = list((zip(x, y)))
    # Find the convex hull
    v = ConvexHull(points).vertices
    # Rotate convex hull vertices until they start from the lowest one
    v = np.roll(v, -v.argmin())
    # Leave only the ascending part
    v = v[:v.argmax()]

    # Create baseline using linear interpolation between vertices
    baseline = np.interp(x, x[v], y[v]).astype('float')
    
    return baseline

def calculate_peak_intensity(wavenumbers, intensities, wavenumber_range):
    """
    Calculate peak intenisty with respect to linear baseline in wavenumber range.
    
    Arguments:
        wavenumbers {array} -- a numpy array 
        intensities {array} -- a numpy array
        wavenumber_range {tuple} -- (low, high)

    Returns:
        peak_wavenumber, peak_relative_intensity {tuple} -- (peak position, peak intensity)
    """
    
    wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    # Find indices corresponding to the wavenumber range
    range_indices = np.where((wavenumbers >= wavenumber_range[0]) & (wavenumbers <= wavenumber_range[1]))

    # Extract the wavenumbers and intensities within the range
    range_wavenumbers = wavenumbers[range_indices]
    range_intensities = intensities[range_indices]

    # Find the index of the peak maximum within the range
    peak_index = np.argmax(range_intensities)
    peak_wavenumber = range_wavenumbers[peak_index]
    peak_intensity = range_intensities[peak_index]

    # Fit a linear baseline to the low and high ends of the wavenumber range
    baseline_indices = np.where((wavenumbers < wavenumber_range[0]) | (wavenumbers > wavenumber_range[1]))
    baseline_wavenumbers = wavenumbers[baseline_indices]
    baseline_intensities = intensities[baseline_indices]
    baseline_slope, baseline_intercept = np.polyfit(baseline_wavenumbers, baseline_intensities, 1)

    # Calculate the intensity of the peak relative to the linear baseline
    peak_intensity_relative = peak_intensity - (baseline_slope * peak_wavenumber + baseline_intercept)

    return peak_wavenumber, peak_intensity_relative


def polynomial_background(wavenumber, absorbance, odr=10, s=0.006, fct='atq'):
    '''
    Python implmentation of V. Mazet's backcorr function originally written in MATLAB.
    Please see https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction
    for detailed explaination and documentation.
    
    Arguments:
        wavenumber {array} -- array of wavenumber values
        absorbance {array} -- array of absorbance values
        odr {int} -- polynomial order
        s = float
        fct = string specifying 'sh', 'ah', 'stq', or 'atq'
    
    Reuturns:
        background {array} -- array of absorbance values corresponding to the fitted background
        a =  numpy array of shape [ord,]
        it = integer
        odr = integer
        s = float
        fct = str
    '''
    N = len(wavenumber) 
    i = np.argsort(wavenumber)
    n = np.sort(wavenumber)
    
    y = absorbance[i]
    
    maxy = max(y)
    
    dely = (maxy - min(y)) / 2
    n = 2 * (n[:] - n[N-1]) / (n[N-1] - n[0]) + 1
    
    y = (y[:] - maxy) / dely + 1
    
    p = np.arange(0,odr+1)
    
    T = np.power(np.matlib.repmat(n, odr+1, 1).transpose(), np.matlib.repmat(p, N, 1))
    
    Tinv = np.matmul(np.linalg.pinv(np.matmul(T.transpose(),T)), T.transpose())
    
    a = np.matmul(Tinv, y)
    z = np.matmul(T, a)
    
    alpha = 0.99 * 1/2
    it = 0               
    zp = np.ones(N)
    
    while sum((z-zp)**2)/sum(zp**2) > 1e-9:
        it = it + 1
        zp = z
        res = y - z
    
        if fct == 'sh':
            d = (res*(2*alpha-1)) * (abs(res)<s) + (-alpha*2*s-res) * (res<=-s) + (alpha*2*s-res) * (res>=s)
        elif fct == 'ah':
            d = (res*(2*alpha-1)) * (res<s) + (alpha*2*s-res) * (res>=s)
        elif fct == 'stq':
            d = (res*(2*alpha-1)) * (abs(res)<s) - res * (abs(res)>=s)
        elif fct == 'atq':
            d = (res*(2*alpha-1)) * (res<s) - res * (res>=s)
        
        a = np.matmul(Tinv, (y+d))  
        z = np.matmul(T,a)
    
    j = np.argsort(i)
    z = (z[j]-1)*dely + maxy

    a[0] = a[0]-1
    a = a*dely
    
    return z, a, it, odr, s, fct

def roundWavenumbers(dataframe):
    """
    Function for rounding the wavenumbers so that there is no mismatch between the machine 
    precision of saving the files in Quasar and the wavenumbers output by the FTIR microscope
    """
    last_nonwavenum_idx = dataframe.columns.get_loc('1981.7 - 2095.8') + 1

    dataframe = dataframe.rename(columns=lambda c: round(float(c), 1) if c not in dataframe.columns[:last_nonwavenum_idx] else c)
    return dataframe

def distribution_Selection(df, distributionIdx, numberOfSigmas):
    """
    Screens data based on the distribution of a given column in a pandas dataframe. 
    Built for screening out dataset values that do not have a sufficient 2019 wavenumber 
    integral for the PEX project. Can be modified to projects liking. 

    Returns the selected datas indexes, the discarded indexes, the mask used to select the data,
    the x position of the distribution mode, and the column values of the distribution that is
    being used to screen data. In this case the values of the baseline integral of the PE 2019
    wavenumber peak. 
    """
    
    df[distributionIdx] = df[distributionIdx].astype(float)

    #creating variable for the array of Polyethylene Area
    area = df['1981.7 - 2095.8'].values

    # Use Gaussian KDE to find the center position of the rightmost mode by creating x coords
    # and then using np.argmax to determine the point with the highest number of counts
    kde = gaussian_kde(area)
    xs = np.linspace(area.min(), area.max(), 1000) 
    modePosition = xs[np.argmax(kde(xs))]

    # Use only the values greater than 1.5 to determine the std of the PE peaks
    positive_vals = area[area > 1.5]
    sigma = positive_vals.std()

    lowerDistributionBound = modePosition - numberOfSigmas * sigma
    upperDistributionBound = modePosition + numberOfSigmas * sigma

    # Select the range of PE normalization integral to be accepted by the mask
    mask_selected = (area >= lowerDistributionBound) & (area <= upperDistributionBound)
    selected_indexes = df.index[mask_selected]
    discarded_indexes = df.index[~mask_selected]

    return selected_indexes, discarded_indexes, mask_selected, modePosition, area