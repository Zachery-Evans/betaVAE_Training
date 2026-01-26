import matplotlib.pyplot as plt
import pandas as pd
import bvae_model as bvae
import tensorflow as tf
import numpy as np
from spectrum_preprocessing import roundWavenumbers, distribution_Selection, pipeline

decoder = tf.saved_model.load("./new_decoder")
encoder = tf.saved_model.load("./new_encoder")

path = './spectral_data/'

file = path + 'SMP65#010 21d 820um.csv'

dataframe = pd.read_csv(file, low_memory=False, skiprows=[1,2])

dataframe = dataframe.sample(frac=0.2, random_state=42).reset_index()

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
    
    frequencies, spectrum = pipeline(frequencies, spectrum)

    interpDataFramelist.append(pd.DataFrame(data=[spectrum], columns=frequencies))

interpRawTrainingDataframe = pd.concat(interpDataFramelist, ignore_index=True)
interpDataFramelist = None # Clear memory

test_array = np.asarray(interpRawTrainingDataframe.values, dtype=np.float32)
interpRawTrainingDataframe = None # Clear memory

z_mean, z_log_var, _ = encoder(test_array, training=False)
print("Z Mean shape:", z_mean.shape)
print("Z Log Var shape:", z_log_var.shape)

latent_log_vars = np.mean(z_log_var, axis=0)

latent_vars = np.exp(latent_log_vars)
for i in range(len(latent_vars)):
    print(f"Latent Variable {i+1} Variance: {latent_vars[i]}")

#print(z_log_var)

"""
plt.figure(figsize=(12, 10))
plt.plot(z_mean[:, 0])
plt.plot(z_mean[:, 1])
plt.xlabel("")
plt.ylabel("")
plt.show()
"""