import matplotlib.pyplot as plt
import pandas as pd
import bvae_model as bvae
import tensorflow as tf
import numpy as np
from spectrum_preprocessing import roundWavenumbers, distribution_Selection, pipeline

decoder = tf.saved_model.load("./new_decoder")
encoder = tf.saved_model.load("./new_encoder")

file =  'interpolated_training_data.csv'

test_array = pd.read_csv(file, header=None).to_numpy().astype('float32')

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