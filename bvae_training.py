import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS BACKEND"] = "tensorflow"
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from spectrum_preprocessing import roundWavenumbers, distribution_Selection, pipeline
import matplotlib.pyplot as plt

"""
Linear Callback for Annealing the Beta Value

"""
class LinearBetaAnneal(keras.callbacks.Callback):
    """
    Anneals the beta variable in betaVAE in a linear motion:
    beta begins at zero, then has a warmup period given by the programmer, 
    then once it reaches the maximum allowed beta value, maintains its value
    for the remainder of the training epochs. 
    
    """
    def __init__(self, vae, warmup_epochs=10, beta_max=15):
        self.vae = vae
        self.warmup_epochs = warmup_epochs
        self.beta_max = beta_max

    def on_epoch_begin(self, epoch, logs=None):
        vae.beta.assign(self.beta_max * min(1.0 , epoch / self.warmup_epochs))


"""
Cyclical Callback for Annealing the Beta Value

"""
class CyclicalBetaAnneal(keras.callbacks.Callback):
    """
    Anneals the beta variable in betaVAE in a sawtooth motion: 
    beta begins at zero, then has a warmup period of (by default) half the cycle length,
    holds the maximum beta for (by default) half of a cycle, and then returns beta to zero to repeat 
    the process as many times as possible for given epochs.  
    
    """
    def __init__(self, vae, cycle_length=20, warmup_ratio=0.5, beta_max=15):
        self.vae = vae
        self.cycle_length = cycle_length
        self.warmup_ratio = warmup_ratio
        self.beta_max = beta_max

    def on_epoch_begin(self, epoch, logs=None):
        cycle_pos = epoch % self.cycle_length
        warmup_epochs = int(self.cycle_length * self.warmup_ratio)
        vae.beta.assign(self.beta_max * min(1.0, cycle_pos / warmup_epochs))
        
"""
Capacity Annealing

"""
class CapacityAnneal(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, c_max, n_steps):
        self.c_max = tf.constant(c_max, dtype=tf.float32)
        self.n_steps = tf.constant(n_steps, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.minimum(self.c_max, self.c_max * step / self.n_steps)


"""
Reparameterization 

"""
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean), seed=1337)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
BetaVAE Model

"""
def calculate_kl_divergence(z_mean, z_logvar):
    # shape: (batch, latent_dim)
    kl_per_dim = 0.5 * (
        tf.square(z_mean) +
        tf.exp(z_logvar) -
        z_logvar - 1.0
    )
    # shape: (batch,)
    kl_per_sample = tf.reduce_sum(kl_per_dim, axis=1)
    # scalar
    kl_mean = tf.reduce_mean(kl_per_sample)

    return kl_mean, kl_per_dim

class BetaVAE(keras.Model):
    def __init__(self, encoder, decoder, beta, c_max, c_steps, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder # ENCODER HAS THE SAMPLING LAYER INSIDE
        self.decoder = decoder
        self.beta = tf.Variable(beta, trainable=False, dtype=tf.float32)

        self.capacity_annealer = CapacityAnneal(c_max=c_max, n_steps=c_steps)
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Trackers for the total, reconstruction, and KL losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def call(self,data):
        z_mean, z_logvar, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction
    
    @tf.function
    def train_step(self, data):

        if isinstance(data, tuple):
            data = data[0]   # keep only x
        
        with tf.GradientTape() as tape:
            
            z_mean, z_logvar, z = self.encoder(data)
            reconstruction = self.decoder(z)

            capacity = self.capacity_annealer(self.step_counter)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction), axis=1) #/ tf.cast(tf.shape(data)[1], tf.float32)
            )

            """
            KL loss per batch vertsion of KL loss

            """
            kl, _ = calculate_kl_divergence(z_mean, z_logvar)

            kl_loss = self.beta * tf.abs(kl - capacity)

            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        self.step_counter.assign_add(1)

        return  {
        "total_loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl,
        "capacity": capacity
        }
    
    def test_step(self, data):

        if isinstance(data, tuple):
            data = data[0]

        # Forward pass
        z_mean, z_logvar, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)

        capacity = self.capacity_annealer(self.step_counter)

        # Reconstruction loss (same as train_step)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(data - reconstruction), axis=1) #/ tf.cast(tf.shape(data)[1], tf.float32)
        )

        """
        KL loss per batch vertsion of KL loss

        """
        kl, _ = calculate_kl_divergence(z_mean, z_logvar)

        kl_loss = tf.abs(kl - capacity)

        total_loss = recon_loss + kl_loss

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

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
    frequencies = interpRawTrainingDataframe.columns.astype(float)

    print("Data preprocessing and interpolation complete.")

else:
    interpRawTrainingDataframe = pd.read_csv("interpolated_training_data.csv", low_memory=False)
    frequencies = interpRawTrainingDataframe.columns.astype(float)

"""
Prepare the training and validation data

"""
wavenumbers = [str(wavenumber) for wavenumber in sorted(frequencies)]
betaVAE_trainingData = interpRawTrainingDataframe[wavenumbers]
betaVAE_validationData = validation_df[wavenumbers]

trainingArray = np.asarray(betaVAE_trainingData.values, dtype=np.float32)
validationArray = np.asarray(betaVAE_validationData.values, dtype=np.float32)

scaler = MinMaxScaler(feature_range=(0,1))

normalizedTrainingArray = scaler.fit_transform(trainingArray)
normalizedValidationArray = scaler.fit_transform(validationArray)

X_train, X_val = train_test_split(trainingArray, train_size=0.8)

interpRawValidationDataframe = None # Clear memory
validation_df = None # Clear memory
betaVAE_trainingData = None # Clear memory
betaVAE_validationData = None # Clear memory

"""
Define the model parameters

"""
input_dim = len(wavenumbers)
output_dim = input_dim

batch = 128

hidden_dim1 = 256
hidden_dim2 = 256

latent_dim = 16

beta = 25

epochs = 100

"""
Build the Encoder

"""
input = keras.Input(shape=input_dim, name='spectra_input') 
x = layers.GaussianNoise(0.05)(input)
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
vae = BetaVAE(encoder, decoder, beta, 100, c_steps=1e8)

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

#vae.fit(trainingArray, trainingArray, epochs=epochs, batch_size=batch, callbacks=[LinearBetaAnneal(vae, warmup_epochs=30, beta_max=beta)])

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=20)
vae.fit(X_train, 
    validation_data=(X_val,), 
    epochs=epochs, 
    batch_size=batch, 
    callbacks=[earlyStopping, 
    LinearBetaAnneal(vae, warmup_epochs=10, beta_max=beta)]
    )

"""
Print the KL divergence for each latent dimension on the validation data

"""
z_mean, z_logvar, _ = encoder(X_train, training=False)
print("Z Mean shape:", z_mean.shape)
print("Z Log Var shape:", z_logvar.shape)

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

    z_mean, z_logvar, _ = saved_encoder(X_train, training=False)
    #z_mean, z_log_var, _ = encoder(test_array, training=False)
    print("Z Mean shape:", z_mean.shape)
    print("Z Log Var shape:", z_logvar.shape)

    kl_per_dim = 0.5 * np.mean(
        z_mean**2 + np.exp(z_logvar) - z_logvar - 1,
        axis = 0
    )

    kl_total = np.sum(kl_per_dim)
    kl_percent = kl_per_dim / kl_total * 100

    for i, kl in enumerate(kl_percent):
        print(f"Latent {i} KL(%): {kl:.3f}")

    indices_of_largest = (np.argsort(kl_per_dim)[-5::])[::-1]

    print("Three largest KL divergences:", indices_of_largest)
    print("Total KL divergence:", kl_total)

    saved_encoder = None
    saved_decoder = None

else:
    print("Failed to load the model.")
