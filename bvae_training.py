import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS BACKEND"] = "tensorflow"
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from spectrum_preprocessing import roundWavenumbers, distribution_Selection, pipeline
import matplotlib.pyplot as plt

"""
Linear Callback for Annealing the Beta Value

"""
class LinearBetaCallback(keras.callbacks.Callback):
    def __init__(self, vae, warmup_epochs, beta_max):
        self.vae = vae
        self.warmup_epochs = warmup_epochs
        self.beta_max = beta_max

    def on_epoch_begin(self, epoch, logs=None):
        self.vae.beta = self.beta_max * min(1.0, epoch / self.warmup_epochs)


"""
Cyclical Callback for Annealing the Beta Value

"""
class CyclicalBetaCallback(keras.callbacks.Callback):
    def __init__(self, vae, cycle_length=20, warmup_ratio=0.5, beta_max=15):
        self.vae = vae
        self.cycle_length = cycle_length
        self.warmup_ratio = warmup_ratio
        self.beta_max = beta_max

    def on_epoch_begin(self, epoch, logs=None):
        cycle_pos = epoch % self.cycle_length
        warmup_epochs = int(self.cycle_length * self.warmup_ratio)
        self.vae.beta = self.beta_max * min(1.0, cycle_pos / warmup_epochs)

"""
Reparameterization 

"""
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
BetaVAE Model

"""
class BetaVAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder # ENCODER HAS THE SAMPLING LAYER INSIDE
        self.decoder = decoder
        self.beta = beta

        # Trackers for the total, reconstruction, and KL losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self,data):
        z_mean, z_logvar, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def train_step(self, data):

        if isinstance(data, tuple):
            data = data[0]   # keep only x

        with tf.GradientTape() as tape:
            
            z_mean, z_logvar, z = self.encoder(data)
            reconstruction = self.decoder(z)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
            )

            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return  {
        "total_loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss,
        }
    
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_logvar, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)

        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(data - reconstruction), axis=1)
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
        )

        total_loss = recon_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta,
        })
        return config

"""
Load and Format the VALIDATION data 

"""
stdDevs = 2
print("Loading and preprocessing validation data...")
if not os.path.exists('./interpolated_validation_data.csv'):
    validation_df = pd.read_csv('./spectral_data/SMP65#013 35d 920um.csv', low_memory=False, skiprows=[1,2])
    validation_df = roundWavenumbers(validation_df)

    last_nonwavenum_idx = validation_df.columns.get_loc('1981.7 - 2095.8') + 1
    wavenumbers = validation_df.columns[last_nonwavenum_idx:]
    
    selected_indexes, discarded_indexes, mask_selected, modePosition, areaPE = distribution_Selection(validation_df, '1981.7 - 2095.8', stdDevs)
    validation_df = validation_df[mask_selected]

    interpRawValidationDataframe = pd.DataFrame()
    interpDataFramelist=[]
    for index, row in validation_df.iterrows():
        row = row[last_nonwavenum_idx:]

        frequencies = row.index.to_numpy()
        frequencies = frequencies[::-1]  # Reverse the order for interpolation
        spectrum = row.to_numpy()
        spectrum = spectrum[::-1]  # Reverse the order for interpolation
        
        interp_frequencies, interp_spectrum = pipeline(frequencies, spectrum)

        interpDataFramelist.append(pd.DataFrame(data=[interp_spectrum], columns=interp_frequencies))

    validation_df = pd.concat(interpDataFramelist, ignore_index=True)
    
    interpDataFramelist = None # Clear memory

    validation_df.to_csv("interpolated_validation_data.csv", index=False)
else:
    validation_df = pd.read_csv("interpolated_validation_data.csv", low_memory=False)


"""
Load and Format the TRAINING data

"""
print("Loading and preprocessing training data...")
if not os.path.exists('./interpolated_training_data.csv'):
    path = './training_data/'
    #List all of the files in the data directory.
    allFiles = os.listdir(path)
    # Take only the files that contain data pertaining to SMP65#010 
    trainingFiles = [file for file in allFiles if file.endswith('.csv') and 'SMP65#010' in file and 'full width' not in file]

    trainingFiles = sorted(trainingFiles, key=lambda x: int(re.search(r'(?<= )(.+?)(?=d)', x).group()))

    # Read all of the data and place the dataframes into a list
    trainingDataframeList = [pd.read_csv(path+file, low_memory=False, skiprows=[1,2]) for file in trainingFiles]

    last_nonwavenum_idx = trainingDataframeList[0].columns.get_loc('1981.7 - 2095.8') + 1
    wavenumbers = trainingDataframeList[0].columns[last_nonwavenum_idx:]
    
    masked_trainingDataframeList = []

    for df in trainingDataframeList:
        selected_indexes, discarded_indexes, mask_selected, modePosition, areaPE = distribution_Selection(df, '1981.7 - 2095.8', stdDevs)

        masked_trainingDataframeList.append(df[mask_selected])
    trainingDataframeList = None # Clear memory

    # Lambda function to round all of the wavenumbers so that column labels are all matching, and then concatenate all the datasets together
    rawTrainingDataframe = pd.concat(
        (
            df.rename(
                columns=lambda c: round(float(c), 1) if c not in df.columns[:last_nonwavenum_idx] else c
            )
            for df in masked_trainingDataframeList
        ),
        ignore_index=True
    )
    masked_trainingDataframeList = None # Clear memory

    interpRawTrainingDataframe = pd.DataFrame()
    interpDataFramelist=[]

    # Interpolate the spectra to increase the accuracy of the model
    for index, row in rawTrainingDataframe.iterrows():

        row = row[last_nonwavenum_idx:]

        frequencies = row.index.to_numpy()
        frequencies = frequencies[::-1]  # Reverse the order for interpolation
        spectrum = row.to_numpy()
        spectrum = spectrum[::-1]  # Reverse the order for interpolation
        
        frequencies, spectrum = pipeline(frequencies, spectrum)

        interpDataFramelist.append(pd.DataFrame(data=[spectrum], columns=frequencies))

    interpRawTrainingDataframe = pd.concat(interpDataFramelist, ignore_index=True)
    interpDataFramelist = None # Clear memory
    interpRawTrainingDataframe.to_csv("interpolated_training_data.csv", index=False)

    print("Data preprocessing and interpolation complete.")

else:
    interpRawTrainingDataframe = pd.read_csv("interpolated_training_data.csv", low_memory=False)
    frequencies = interpRawTrainingDataframe.columns.astype(float)


"""
K-fold Validation

*** Need to rework this section to properly implement k-fold validation for the interpolated data ***

unique_samples = rawTrainingDataframe["Sample Name"].unique()
n_folds = 5

fold_map = {}
for i, sid in enumerate(unique_samples):
    fold_map[sid] = i % n_folds

rawTrainingDataframe["fold"] = rawTrainingDataframe["Sample Name"].map(fold_map)
"""


"""
Initialize the model, train the model and save. 
"""
wavenumbers = [str(wavenumber) for wavenumber in sorted(frequencies)]
betaVAE_trainingData = interpRawTrainingDataframe[wavenumbers]
betaVAE_validationData = validation_df[wavenumbers]
input_dim = len(wavenumbers)
output_dim = input_dim

batch = 64

hidden_dim1 = 128
hidden_dim2 = 64

latent_dim = 16

beta = 50

epochs = 300

trainingArray = np.asarray(betaVAE_trainingData.values, dtype=np.float32)
validationArray = np.asarray(betaVAE_validationData.values, dtype=np.float32)

interpRawValidationDataframe = None # Clear memory
validation_df = None # Clear memory
betaVAE_trainingData = None # Clear memory
betaVAE_validationData = None # Clear memory

"""
Build the Encoder

"""
input = keras.Input(shape=input_dim, name='spectra_input') 
x = layers.Dense(hidden_dim1, activation='relu')(input) 
x = layers.Dense(hidden_dim2, activation='relu')(x) 
z_mean = layers.Dense(latent_dim, name='z_mean')(x) 
z_logvar = layers.Dense(latent_dim, name='z_logvar')(x) 
z_logvar = tf.clip_by_value(z_logvar, -10, 10)
z = Sampling()([z_mean, z_logvar])
encoder = keras.Model(inputs=input, outputs=[z_mean, z_logvar, z], name='encoder')
encoder.summary()

"""
Build the Decoder 

"""
latent_inputs = keras.Input(shape=(latent_dim,), name='latent_variables')
x = layers.Dense(hidden_dim2, activation='relu')(latent_inputs)
x = layers.Dense(hidden_dim1, activation='relu')(x)
outputs = layers.Dense(output_dim, activation='linear')(x)
decoder = keras.Model([latent_inputs], outputs, name='decoder')
decoder.summary()

"""
Build the VAE Model and Train

"""
vae = BetaVAE(encoder, decoder, beta)

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

vae.fit(trainingArray, epochs=epochs, batch_size=batch, callbacks=[LinearBetaCallback(vae, warmup_epochs=30, beta_max=beta)])

"""
Print the KL divergence for each latent dimension on the validation data

"""
z_mean, z_log_var, _ = encoder(trainingArray, training=False)
#z_mean, z_log_var, _ = encoder(test_array, training=False)
print("Z Mean shape:", z_mean.shape)
print("Z Log Var shape:", z_log_var.shape)

kl_per_dim = 0.5 * np.mean(
    z_mean**2 + np.exp(z_log_var) - z_log_var - 1,
    axis=0
)

for i, kl in enumerate(kl_per_dim):
    print(f"Latent {i+1} KL: {kl:.3f}")

indices_of_largest = np.argsort(kl_per_dim)[-3:]

print("Three largest KL divergences:", indices_of_largest)


"""
Save the model and reload to test that it saved correctly

"""

#tf.saved_model.save(vae, "./new_vae/")
tf.saved_model.save(encoder, './new_encoder/')
tf.saved_model.save(decoder, './new_decoder/')

encoder = None # Clear memory
decoder = None # Clear memory

saved_encoder = tf.saved_model.load("./new_encoder/")
saved_decoder = tf.saved_model.load("./new_decoder/")
#saved_vae = tf.saved_model.load("./new_vae/")

if saved_encoder is not None and saved_decoder is not None:
    print("All models loaded successfully!")

    z_mean, z_log_var, _ = saved_encoder(trainingArray, training=False)
    #z_mean, z_log_var, _ = encoder(test_array, training=False)
    print("Z Mean shape:", z_mean.shape)
    print("Z Log Var shape:", z_log_var.shape)

    kl_per_dim = 0.5 * np.mean(
        z_mean**2 + np.exp(z_log_var) - z_log_var - 1,
        axis=0
    )

    for i, kl in enumerate(kl_per_dim):
        print(f"Latent {i} KL: {kl:.3f}")

    saved_encoder = None
    saved_decoder = None

else:
    print("Failed to load the model.")
