import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import bvae_model as bvae
import tensorflow as tf
import numpy as np

decoder = tf.saved_model.load("./new_decoder")
encoder = tf.saved_model.load("./new_encoder")

file =  'training_data.csv'

df = pd.read_csv(file)

frequencies = df.columns.astype(float)

def encode(X_np, batch=4096):
    z_means = []
    for i in range(0, len(X_np), batch):
        zm, zl, z = encoder(X_np[i:i+batch])
        z_means.append(zm)
    return np.vstack(z_means)

def reconstruct(X_np, batch=4096):
    Xh = []
    for i in range(0, len(X_np), batch):
        zm, zl, z = encoder(X_np[i:i+batch])
        xh = decoder(zm)
        Xh.append(xh)
    return np.vstack(Xh)

X = df.values.astype(np.float32)
# --- Encode all data deterministically ---
Z = encode(X)  # shape (n_samples, LATENT_DIM)

kept_wn = df.columns.values.astype('float')

# pick random examples to visualize
N_SHOW = 5
rng = np.random.default_rng(42)
idxs = rng.choice(len(X), size=min(N_SHOW, len(X)), replace=False)

X_subset = X[idxs]
X_recon = reconstruct(X_subset)

wn = np.array(kept_wn)

plt.figure(figsize=(10, 12))
for i, (x_true, x_rec) in enumerate(zip(X_subset, X_recon), start=1):
    plt.subplot(N_SHOW, 1, i)
    plt.plot(wn, x_true, color="black", lw=1.5, label="Experimental")
    plt.plot(wn, x_rec, color="red", lw=1.0, alpha=0.8, label="Reconstruction")
    plt.gca().invert_xaxis()
    if i == 1:
        plt.legend(fontsize=9)
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Spectrum {i}", fontsize=9)
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.tight_layout()
plt.savefig("BVae_example_reconstructions.jpeg")
plt.show()


X = df.values.astype(np.float32)
# --- Encode all data deterministically ---
Z = encode(X)  # shape (n_samples, LATENT_DIM)

# --- Rank dimensions by variance ---
variances = np.var(Z, axis=0)
ranked_dims = np.argsort(variances)[::-1]
print("Latent dimensions ranked by variance (most informative first):")
for i, dim in enumerate(ranked_dims):
    print(f"  Dim {dim}  Var={variances[dim]:.4f}")

# --- Parameters for traversal ---
N_STEPS = 7                # how many points to sample per dim
SIGMA_SCALE = 3.0          # ± range to explore
TOP_N = 8 # how many top dims to visualize
wn = np.array(kept_wn)

# --- Compute the latent mean for reference ---
z_mean_global = np.mean(Z, axis=0)

# Colormap
cmap = plt.get_cmap("viridis")

# --- Traverse each top dimension individually ---
for d in ranked_dims[:TOP_N]:
    z_ref = np.copy(z_mean_global)
    std_d = np.std(Z[:, d])
    traversal_values = np.linspace(
        z_ref[d] - SIGMA_SCALE * std_d,
        z_ref[d] + SIGMA_SCALE * std_d,
        N_STEPS
    )

    traversed_spectra = []
    for val in traversal_values:
        z_temp = np.copy(z_ref)
        z_temp[d] = val
        x_dec = decoder(z_temp[np.newaxis, :])
        traversed_spectra.append(x_dec[0])
    traversed_spectra = np.array(traversed_spectra)

    # --- Plot with viridis colors ---
    plt.figure(figsize=(8, 4))

    for i, (spec, val) in enumerate(zip(traversed_spectra, traversal_values)):
        # map i -> [0,1] for colormap
        if N_STEPS > 1:
            t = 0.8*(i / (N_STEPS - 1))
        else:
            t = 0.5
        color = cmap(t)

        plt.plot(
            wn,
            spec,
            label=f"{val:.2f}",
            color=color,
            lw=1.5
        )

    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Latent dim {d} traversal (σ={std_d:.3f})")
    plt.legend(title="z value", fontsize=8)
    plt.tight_layout()
    #plt.savefig(f"Zero_offset_Latent_dim_{d}_traversal_sigma_{std_d:.3f}.jpeg",
    #            dpi=300)
    plt.show()

"""
# display a 2D plot of the digit classes in the latent space
z_mean, _, _ = encoder(data)
plt.figure(figsize=(12, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1])
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()
"""