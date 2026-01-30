import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
import pandas as pd
import bvae_model as bvae
import tensorflow as tf
import numpy as np
from spectrum_preprocessing import roundWavenumbers, distribution_Selection, pipeline

decoder = tf.saved_model.load("./new_decoder")
encoder = tf.saved_model.load("./new_encoder")

file =  'interpolated_training_data.csv'

test_df = pd.read_csv(file)

wavenumbers = test_df.columns

frequencies = sorted(wavenumbers.astype(float))

test_array = np.asarray(test_df.values, dtype=np.float32)

def plot_label_clusters(encoder, data, labels=None):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c= labels, cmap='viridis')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

plot_label_clusters(encoder, test_array)

def plot_latent_traversal(decoder, encoder, data, latent_dim, n_steps=10, range_min=-3, range_max=3):
    # Select a random sample from the data
    sample = data
    z_mean, _, _ = encoder(sample)
    z_mean = z_mean.numpy()

    # Create a figure to hold the subplots
    fig, axes = plt.subplots(latent_dim, figsize=(n_steps * 2, latent_dim * 2))
    
    # Generate values to traverse dimensions
    traversal_values = np.linspace(range_min, range_max, n_steps)

    # For each latent dimension
    for dim in range(latent_dim):
        for i, val in enumerate(traversal_values):
            # Create a copy of the mean latent vector
            z_traversal = np.array(z_mean, copy=True)
            z_traversal[0, dim] = val

            # Decode the modified latent vector
            reconstructed = decoder(tf.convert_to_tensor(z_traversal, dtype=tf.float32))
            reconstructed = np.mean(reconstructed, axis=0)

            # Plot the reconstructed spectrum
            if dim < latent_dim-1:
                axes[dim].plot(frequencies, reconstructed)
                axes[dim].set_title(f"Dim {dim+1} Traversal")
                axes[dim].set_yticks([])
                axes[dim].set_xticks([])

            else:
                axes[dim].plot(frequencies[::-1], reconstructed[::-1])
                axes[dim].set_title(f"Dim {dim+1} Traversal")
                axes[dim].set_yticks([])

    plt.show()


plot_latent_traversal(decoder, encoder, test_array, latent_dim=3, n_steps=10, range_min=-1e6, range_max=1e6)