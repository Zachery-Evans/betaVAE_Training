import matplotlib.pyplot as plt
import pandas as pd
import bvae_model as bvae
import tensorflow as tf
import numpy as np
from spectrum_preprocessing import roundWavenumbers, distribution_Selection

decoder = tf.saved_model.load("./new_decoder")
encoder = tf.saved_model.load("./new_encoder")

path = './training_data/'

file = path + 'SMP65#010 70d 820um.csv'

dataframe = pd.read_csv(file, skiprows=[1,2])

selected_indexes, discarded_indexes, mask_selected, modePosition, area = distribution_Selection(dataframe, '1981.7 - 2095.8', 3)
dataframe = dataframe[mask_selected]
dataframe = roundWavenumbers(dataframe)

last_nonwavenum_idx = dataframe.columns.get_loc('1981.7 - 2095.8') + 1

interpDataFramelist = []
for index, row in dataframe.iterrows():

    row = row[last_nonwavenum_idx:]

    frequencies = row.index.to_numpy()
    frequencies = frequencies[::-1]  # Reverse the order for interpolation
    spectrum = row.to_numpy()
    spectrum = spectrum[::-1]  # Reverse the order for interpolation
    
    frequencies, spectrum = bvae.pipeline(frequencies, spectrum)

    interpDataFramelist.append(pd.DataFrame(data=[spectrum], columns=frequencies))

interpRawTrainingDataframe = pd.concat(interpDataFramelist, ignore_index=True)
interpDataFramelist = None # Clear memory

dataframe = dataframe[dataframe.columns[dataframe.columns.get_loc('1981.7 - 2095.8'):]]
array = np.asarray(dataframe.values, dtype=np.float32)

# display a 2D plot of the digit classes in the latent space
z_mean, _= encoder.predict(array, verbose=0)

print(z_mean.shape)
"""
plt.figure(figsize=(12, 10))
plt.plot(z_mean[:, 0])
plt.plot(z_mean[:, 1])
plt.xlabel("")
plt.ylabel("")
plt.show()
"""