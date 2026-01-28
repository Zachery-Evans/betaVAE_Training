"""
This file contains functions for implementing beta-VAE model on new experimental IR spectra
Dependencies: pandas, numpy, matplotlib, sklearn, tensorflow
Author: Michael Grossutti
Date Created: 2023-02-03
"""

#imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras
from keras import Model
from keras import layers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import spectrum_preprocessing as sp

#load the saved encoder, and decoder

loaded_decoder = tf.saved_model.load("./new_decoder")
loaded_encoder = tf.saved_model.load("./new_encoder")

def encoder(normalized_spectrum_array):
    """
    Encode a PEX-a pipe IR spectrum array to learned latent representations.
    Positive values indicate presence of generative physicochemical factor relative to virgin pipe.
    Negative values indicate absence of generative physicochemical factor relative to virgin pipe.
    
    Arguments:
        spectrum_array {array} -- An array corresponding to the interpolated (1 cm-1 sapcing) normalized absorbance values 
                                of an IR spectrum. 
                                The array must correspond to 584 absorbance values in two wavenumber windows
                                window1: 898-1200 cm-1, consisting of 303 absorbance points, one at each integer cm-1
                                window2: 1520-1800 cm-1, consisting of 281 absorbance points, one at each integer cm-1
        
    Returns:
        latent_units {array} -- An array with L1, L2, L3 encodings for visualization and mapping to IR images.
        
    """
    normalized_spectrum_array = normalized_spectrum_array.astype('float32')
    encoded_spectrum = loaded_encoder([normalized_spectrum_array])
    L1 = 1*np.array(encoded_spectrum[0])[0][1]
    L2 = 1*np.array(encoded_spectrum[0])[0][3]
    L3 = -1*np.array(encoded_spectrum[0])[0][2]
    
    return L1,L2,L3

def reconstruction_MSE(normalized_spectrum_array):
    """"
    Reconstructs an experimental IR spectrum using the trained encoder-decoder and calculates MSE.
    
    Arguments:
        normalized_spectrum_array {array} -- An array corresponding to the normalized absorbance values of an IR spectrum. 
                                 The array must correspond to 584 absorbance values in two wavenumber windows
                                window1: 898-1200 cm-1, consisting of 303 absorbance points, one at each integer cm-1
                                window2: 1520-1800 cm-1, consisting of 281 absorbance points, one at each integer cm-1
    
     Returns:
        MSE {float} -- The mean squared error (MSE) of the reconstructed spectrum compared to experimental spectrum
    """
    normalized_spectrum_array = normalized_spectrum_array.astype('float32')
    encoded_spectrum = loaded_encoder([normalized_spectrum_array])
    encoded_spectrum = np.array(encoded_spectrum[0])[0]
    decoded_spectrum = loaded_decoder([encoded_spectrum])[0]
    MSE = mean_squared_error(normalized_spectrum_array, decoded_spectrum)
    
    return MSE


def reconstruction_plot(normalized_spectrum_array):
    """
    Reconstructs an experimental IR spectrum using the trained encoder-decoder and plots it alongisde the input spectrum.
    
    Arguments:
        normalized_spectrum_array {array} -- An array corresponding to the normalized absorbance values of an IR spectrum. 
                                The array must correspond to 584 absorbance values in two wavenumber windows
                                window1: 898-1200 cm-1, consisting of 303 absorbance points, one at each integer cm-1
                                window2: 1520-1800 cm-1, consisting of 281 absorbance points, one at each integer cm-1
    
    Returns:
        Plot -- Black line is experimental spectrum, red line is reconstructed spectrum.
    """
    normalized_spectrum_array = normalized_spectrum_array.astype('float32')
    encoded_spectrum = loaded_encoder([normalized_spectrum_array])
    encoded_spectrum = np.array(encoded_spectrum[0])[0]
    decoded_spectrum = loaded_decoder([encoded_spectrum])[0]
    
    return plt.plot(normalized_spectrum_array, c='k'), plt.plot(decoded_spectrum, c='r')

def reconstructed_spectrum(normalized_spectrum_array):
    """
    Reconstructs an experimental IR spectrum using the trained encoder-decoder and plots it alongisde the input spectrum.
    
    Arguments:
        normalized_spectrum_array {array} -- An array corresponding to the normalized absorbance values of an IR spectrum. 
                                The array must correspond to 584 absorbance values in two wavenumber windows
                                window1: 898-1200 cm-1, consisting of 303 absorbance points, one at each integer cm-1
                                window2: 1520-1800 cm-1, consisting of 281 absorbance points, one at each integer cm-1
    
    Returns:
        Plot -- Black line is experimental spectrum, red line is reconstructed spectrum.
    """
    normalized_spectrum_array = normalized_spectrum_array.astype('float32')
    encoded_spectrum = loaded_encoder([normalized_spectrum_array])
    encoded_spectrum = np.array(encoded_spectrum[0])[0]
    decoded_spectrum = loaded_decoder([encoded_spectrum])[0]
    
    return decoded_spectrum

def bvae_pipeline(expt_wavenumber, expt_absorbance):
    f = expt_wavenumber
    a = expt_absorbance

    wavenumber, normalized_absorbance = sp.pipeline(f, a)

    reconstructed = reconstructed_spectrum(normalized_absorbance)
    mse = reconstruction_MSE(normalized_absorbance)
    encodings = encoder(normalized_absorbance)
    
    return (wavenumber, normalized_absorbance, encodings, mse, np.asarray(reconstructed))

